#include "llmserve/engine.h"

#include "llmserve/block_pool.h"
#include "llmserve/prefix_index.h"
#include "llmserve/scheduler.h"
#include "llmserve/utf8.h"

#include <algorithm>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace llmserve {
namespace {

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::int64_t age_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

EngineConfig checked_config(EngineConfig config) {
    config.validate();
    return config;
}

} // namespace

RequestHandle::RequestHandle(RequestInput input, std::vector<Token> prompt,
                             std::size_t capacity, std::uint64_t order,
                             Clock::time_point created)
    : input_(std::move(input)), prompt_(std::move(prompt)), capacity_(capacity),
      order_(order), created_(created),
      deadline_(created + std::chrono::milliseconds(input_.timeout_ms)) {}

std::optional<Event> RequestHandle::next(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    cv_.wait_for(lock, timeout, [&] { return !events_.empty() || finished_; });
    if (events_.empty()) {
        return std::nullopt;
    }
    auto event = std::move(events_.front());
    events_.pop_front();
    return event;
}

bool RequestHandle::emit(Event event) {
    std::lock_guard lock(mutex_);
    if (finished_) {
        return false;
    }
    const auto terminal = event.kind != Event::Kind::token;
    if (!terminal && events_.size() >= capacity_) {
        return false;
    }
    events_.push_back(std::move(event));
    finished_ = terminal;
    cv_.notify_all();
    return true;
}

bool RequestHandle::finished() const {
    std::lock_guard lock(mutex_);
    return finished_;
}

struct Engine::Impl {
    struct Active {
        std::shared_ptr<RequestHandle> request;
        SequenceId sequence;
        std::vector<BlockId> blocks;
        std::size_t processed = 0;
        std::size_t generated = 0;
        std::size_t reused = 0;
        Token last_token = 0;
        Utf8Buffer text;
        Clock::time_point started;
        Clock::time_point last_scheduled;
        std::optional<Clock::time_point> first_token;
        bool done = false;
    };

    EngineConfig config;
    std::unique_ptr<ModelRunner> runner;
    BackendCapabilities capabilities;
    BlockPool pool;
    PrefixIndex prefixes;
    mutable std::mutex mutex;
    mutable std::mutex tokenizer_mutex;
    std::condition_variable cv;
    std::atomic<bool> stopping{false};
    bool failed = false;
    bool published = false;
    std::thread worker;
    std::deque<std::shared_ptr<RequestHandle>> incoming;
    std::unordered_map<std::string, std::shared_ptr<RequestHandle>> registry;
    std::vector<std::shared_ptr<RequestHandle>> waiting;
    std::vector<std::unique_ptr<Active>> active;
    std::vector<SequenceId> free_sequences;
    std::vector<SequenceId> free_cache_sequences;
    std::uint64_t next_order = 0;
    std::uint64_t accepted = 0;
    std::uint64_t rejected = 0;
    Statistics counters;
    Statistics snapshot;
    TelemetryCapture capture;
    Clock::time_point telemetry_epoch{};
    std::uint64_t next_batch_id = 0;

    Impl(EngineConfig cfg, std::unique_ptr<ModelRunner> model)
        : config(checked_config(cfg)), runner(std::move(model)),
          pool(config.context_tokens / config.block_size), prefixes(config.block_size) {
        if (!runner || runner->info().context_tokens < config.context_tokens) {
            throw std::invalid_argument("model runner does not satisfy the context capacity");
        }
        capabilities = runner->capabilities();
        if ((capabilities.max_sequences &&
             config.max_active + config.prefix_cache_entries > capabilities.max_sequences) ||
            (capabilities.max_batch_tokens && config.batch_tokens > capabilities.max_batch_tokens) ||
            (capabilities.max_model_len && config.max_model_len > capabilities.max_model_len)) {
            throw std::invalid_argument("Engine 配置超过 model runner 的实际容量");
        }
        if (!capabilities.prefix_copy && (config.prefix_cache_entries || config.prefix_cache_tokens)) {
            throw std::invalid_argument("model runner 不支持 prefix copy/cache");
        }
        if (!capabilities.synchronous_execute) {
            throw std::invalid_argument("Engine 需要同步完成的 model runner");
        }
        capture.mode = config.telemetry_mode;
        if (capture.mode != TelemetryMode::off) {
            capture.batches.resize(config.telemetry_capacity);
            for (auto& batch : capture.batches) {
                batch.slices.resize(config.max_active);
                capture.storage_bytes += batch.slices.capacity() * sizeof(SliceTelemetry);
            }
            capture.storage_bytes += capture.batches.capacity() * sizeof(BatchTelemetry);
            telemetry_epoch = Clock::now();
        }
        for (std::size_t i = config.max_active; i > 0; --i) {
            free_sequences.push_back(static_cast<SequenceId>(i - 1));
        }
        for (std::size_t i = config.prefix_cache_entries; i > 0; --i) {
            free_cache_sequences.push_back(static_cast<SequenceId>(config.max_active + i - 1));
        }
        snapshot.kv_total_blocks = pool.capacity();
        worker = std::thread([this] { run(); });
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return published; });
    }

    ~Impl() { stop(); }

    void stop() {
        stopping.store(true);
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    void publish() {
        counters.active_requests = active.size();
        counters.kv_total_blocks = pool.capacity();
        counters.kv_used_blocks = pool.used_count();
        std::unordered_set<BlockId> active_blocks;
        for (const auto& item : active) {
            if (!item->done) {
                active_blocks.insert(item->blocks.begin(), item->blocks.end());
            }
        }
        counters.kv_active_unique_blocks = active_blocks.size();
        counters.prefix_entries = prefixes.size();
        counters.prefix_tokens = prefixes.token_count();
        counters.resources = runner->resources();
        counters.resources_batch_id = next_batch_id;
        std::lock_guard lock(mutex);
        snapshot = counters;
        published = true;
        cv.notify_all();
    }

    void terminal(const std::shared_ptr<RequestHandle>& request, Event event) {
        if (request->finished()) {
            return;
        }
        if (event.kind == Event::Kind::done) {
            ++counters.completed;
        } else if (event.error_code == "cancelled") {
            ++counters.cancelled;
        } else if (event.error_code == "timeout") {
            ++counters.timed_out;
        } else {
            ++counters.failed;
        }
        {
            std::lock_guard lock(mutex);
            registry.erase(request->id());
        }
        request->emit(std::move(event));
    }

    Event terminal_event(const std::shared_ptr<RequestHandle>& request,
                         const std::string& reason, int status,
                         const std::string& message) const {
        Event event;
        event.kind = status == 200 ? Event::Kind::done : Event::Kind::error;
        event.finish_reason = status == 200 ? reason : "error";
        event.error_code = status == 200 ? "" : reason;
        event.error_message = message;
        event.status = status;
        event.usage.prompt_tokens = request->prompt_.size();
        event.timings.total_ms = milliseconds(request->created_, Clock::now());
        event.timings.queue_ms = event.timings.total_ms;
        return event;
    }

    void require_reusable() const {
        const auto resources = runner->resources();
        if (resources && (!resources->state_valid || !resources->reusable)) {
            throw std::runtime_error("model runner 状态无效或不可复用，Engine 已停止");
        }
    }

    void finish(Active& item, const std::string& reason, int status = 200,
                const std::string& message = "", bool check_backend = true) {
        if (item.done) {
            return;
        }
        runner->clear_sequence(item.sequence);
        if (check_backend) { require_reusable(); }
        pool.release(item.blocks);
        item.blocks.clear();
        free_sequences.push_back(item.sequence);
        item.done = true;
        auto event = terminal_event(item.request, reason, status, message);
        event.text = item.text.finish();
        event.usage.completion_tokens = item.generated;
        event.usage.cached_tokens = item.reused;
        event.timings.queue_ms = milliseconds(item.request->created_, item.started);
        if (item.first_token) {
            event.timings.ttft_ms = milliseconds(item.request->created_, *item.first_token);
        }
        terminal(item.request, std::move(event));
    }

    void erase_finished() {
        std::erase_if(active, [](const auto& item) { return item->done; });
    }

    bool expired(const std::shared_ptr<RequestHandle>& request) const {
        return request->cancelled_.load() || Clock::now() >= request->deadline_;
    }

    void expire_requests() {
        std::erase_if(waiting, [&](const auto& request) {
            if (!expired(request)) {
                return false;
            }
            const auto cancel = request->cancelled_.load();
            terminal(request, terminal_event(request, cancel ? "cancelled" : "timeout",
                cancel ? 409 : 408, cancel ? "request cancelled" : "request deadline exceeded"));
            return true;
        });
        for (auto& item : active) {
            if (expired(item->request)) {
                const auto cancel = item->request->cancelled_.load();
                finish(*item, cancel ? "cancelled" : "timeout", cancel ? 409 : 408,
                    cancel ? "request cancelled" : "request deadline exceeded");
            }
        }
        erase_finished();
    }

    void evict(SequenceId sequence, bool check_backend = true) {
        auto entry = prefixes.erase(sequence);
        runner->clear_sequence(sequence);
        pool.release(entry.blocks);
        free_cache_sequences.push_back(sequence);
        ++counters.cache_evictions;
        if (check_backend) { require_reusable(); }
    }

    bool admit(const std::shared_ptr<RequestHandle>& request) {
        const auto total_blocks =
            (request->prompt_.size() + request->input_.max_tokens + config.block_size - 1) /
            config.block_size;
        auto match = prefixes.match(request->input_.cache_namespace, request->prompt_,
                                     request->prompt_.size() - 1);
        auto shared_blocks = match ? match->tokens / config.block_size : 0;
        while (total_blocks - shared_blocks > pool.free_count() && prefixes.size() > 0) {
            evict(*prefixes.least_recently_used());
            match = prefixes.match(request->input_.cache_namespace, request->prompt_,
                                    request->prompt_.size() - 1);
            shared_blocks = match ? match->tokens / config.block_size : 0;
        }
        auto fresh = pool.allocate(total_blocks - shared_blocks);
        if (!fresh) {
            return false;
        }
        auto item = std::make_unique<Active>();
        item->request = request;
        item->sequence = free_sequences.back();
        free_sequences.pop_back();
        item->blocks.reserve(total_blocks);
        if (match) {
            const auto& source = prefixes.at(match->sequence);
            item->blocks.assign(source.blocks.begin(), source.blocks.begin() +
                                static_cast<std::ptrdiff_t>(shared_blocks));
            pool.retain(item->blocks);
            item->processed = item->reused = match->tokens;
            ++counters.cache_hits;
            counters.cached_tokens += match->tokens;
        }
        item->blocks.insert(item->blocks.end(), fresh->begin(), fresh->end());
        item->started = Clock::now();
        item->last_scheduled = request->created_;
        // 先发布所有权，后端调用失败时即可回收容量预留。
        active.push_back(std::move(item));
        runner->clear_sequence(active.back()->sequence);
        require_reusable();
        if (match) {
            runner->copy_sequence(match->sequence, active.back()->sequence, match->tokens);
        }
        return true;
    }

    void admit_waiting() {
        const auto now = Clock::now();
        std::stable_sort(waiting.begin(), waiting.end(), [&](const auto& left, const auto& right) {
            const auto lp = effective_priority(left->input_.priority, age_ms(left->created_, now), config.aging_ms);
            const auto rp = effective_priority(right->input_.priority, age_ms(right->created_, now), config.aging_ms);
            return lp != rp ? lp > rp : left->order_ < right->order_;
        });
        for (std::size_t i = 0; i < waiting.size() && !free_sequences.empty();) {
            if (admit(waiting[i])) {
                waiting.erase(waiting.begin() + static_cast<std::ptrdiff_t>(i));
            } else if (age_ms(waiting[i]->created_, now) >= config.admission_reserve_ms) {
                break;
            } else {
                ++i;
            }
        }
    }

    void cache_prompt(Active& item) {
        if (config.prefix_cache_entries == 0) {
            return;
        }
        const auto& request = item.request;
        const auto length = std::min(request->prompt_.size(), config.prefix_cache_tokens) /
            config.block_size * config.block_size;
        if (length == 0) {
            return;
        }
        const auto existing = prefixes.match(request->input_.cache_namespace, request->prompt_, length);
        if (existing && existing->tokens == length) {
            return;
        }
        while (prefixes.size() > 0 &&
               (free_cache_sequences.empty() || prefixes.token_count() + length > config.prefix_cache_tokens)) {
            evict(*prefixes.least_recently_used());
        }
        const auto sequence = free_cache_sequences.back();
        free_cache_sequences.pop_back();
        PrefixEntry entry{sequence, request->input_.cache_namespace,
            {request->prompt_.begin(), request->prompt_.begin() + static_cast<std::ptrdiff_t>(length)},
            {item.blocks.begin(), item.blocks.begin() + static_cast<std::ptrdiff_t>(length / config.block_size)}};
        pool.retain(entry.blocks);
        prefixes.insert(std::move(entry));
        runner->clear_sequence(sequence);
        require_reusable();
        runner->copy_sequence(item.sequence, sequence, length);
    }

    std::uint64_t elapsed_ns(Clock::time_point time) const noexcept {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(time - telemetry_epoch).count());
    }

    void iteration(Clock::time_point start, Clock::time_point admitted) {
        const auto batch_id = ++next_batch_id;
        BatchTelemetry* record = nullptr;
        if (capture.mode != TelemetryMode::off) {
            if (capture.recorded < capture.batches.size()) {
                record = &capture.batches[capture.recorded++];
                record->batch_id = batch_id;
                record->start_ns = elapsed_ns(start);
                record->admission_ns = elapsed_ns(admitted) - record->start_ns;
                record->waiting_requests = waiting.size();
                record->active_requests = active.size();
                record->reserved_unique_blocks = pool.capacity() - pool.free_count();
            } else {
                ++capture.dropped;
            }
        }
        struct FinishCapture {
            Impl& engine;
            BatchTelemetry* record;
            ~FinishCapture() {
                if (record) { record->finish_ns = engine.elapsed_ns(Clock::now()); }
            }
        } finish_capture{*this, record};
        const auto now = Clock::now();
        std::vector<ScheduleItem> items;
        for (std::size_t i = 0; i < active.size(); ++i) {
            const auto& item = *active[i];
            items.push_back({i, item.request->prompt_.size() - item.processed,
                item.request->input_.priority, age_ms(item.last_scheduled, now), item.request->order_});
        }
        const auto plan = schedule_batch(items, config);
        const auto scheduled = record ? Clock::now() : Clock::time_point{};
        if (plan.slices.empty()) {
            throw std::logic_error("scheduler made no progress");
        }
        std::vector<BatchToken> tokens;
        tokens.reserve(plan.token_count());
        for (const auto& slice : plan.slices) {
            auto& item = *active[slice.key];
            const auto& prompt = item.request->prompt_;
            if (slice.prefill) {
                for (std::size_t i = item.processed; i < item.processed + slice.tokens; ++i) {
                    tokens.push_back({prompt[i], static_cast<std::int32_t>(i), item.sequence,
                                      i + 1 == prompt.size()});
                }
            } else {
                tokens.push_back({item.last_token,
                    static_cast<std::int32_t>(prompt.size() + item.generated - 1), item.sequence, true});
            }
        }
        if (record) {
            record->scheduler_ns = elapsed_ns(scheduled) - elapsed_ns(admitted);
            record->prefill_tokens = plan.prefill_tokens;
            record->decode_tokens = plan.decode_tokens;
            record->sequences = plan.slices.size();
            for (std::size_t i = 0; i < plan.slices.size(); ++i) {
                const auto& slice = plan.slices[i];
                const auto& item = *active[slice.key];
                auto& observed = record->slices[i];
                std::copy(item.request->id().begin(), item.request->id().end(), observed.request_id.begin());
                observed.request_order = item.request->order_;
                observed.sequence = item.sequence;
                observed.prefill = slice.prefill;
                observed.tokens = slice.tokens;
                observed.context_before = slice.prefill ? item.processed :
                    item.request->prompt_.size() + item.generated - 1;
                observed.logits_tokens = !slice.prefill ||
                    item.processed + slice.tokens == item.request->prompt_.size() ? 1 : 0;
                observed.token_index = item.generated;
                record->logits_tokens += observed.logits_tokens;
                record->context_before_sum += observed.context_before;
                record->context_before_max = std::max(record->context_before_max, observed.context_before);
                const auto after = observed.context_before + observed.tokens;
                record->context_after_sum += after;
                record->context_after_max = std::max(record->context_after_max, after);
            }
            record->resources_before = runner->resources();
            record->runner_start_ns = elapsed_ns(Clock::now());
            record->prepare_ns = record->runner_start_ns - elapsed_ns(scheduled);
        }
        std::vector<Sample> samples;
        try {
            samples = record && capture.mode == TelemetryMode::stages ?
                runner->execute_profiled(tokens, record->runner) : runner->execute(tokens);
            require_reusable();
            if (record) { record->runner_completed = true; }
        } catch (...) {
            if (record) {
                record->runner_ns = elapsed_ns(Clock::now()) - record->runner_start_ns;
                record->resources_after = runner->resources();
            }
            throw;
        }
        if (record) {
            record->runner_ns = elapsed_ns(Clock::now()) - record->runner_start_ns;
            record->resources_after = runner->resources();
        }
        const auto expected_samples = static_cast<std::size_t>(std::count_if(
            tokens.begin(), tokens.end(), [](const auto& token) { return token.logits; }));
        if (samples.size() != expected_samples) {
            throw std::runtime_error("model runner returned an unexpected number of samples");
        }
        // 在发布任何 token 前验证整批结果，后续条目出错时不会泄露本批的部分输出。
        std::vector<bool> expected(config.max_active, false), sampled(config.max_active, false);
        for (const auto& token : tokens) {
            if (token.logits) {
                const auto sequence = static_cast<std::size_t>(token.sequence);
                if (expected[sequence]) { throw std::logic_error("Engine 每个 sequence 每轮最多一个输出"); }
                expected[sequence] = true;
            }
        }
        for (const auto& sample : samples) {
            if (sample.sequence < 0 || static_cast<std::size_t>(sample.sequence) >= expected.size() ||
                !expected[static_cast<std::size_t>(sample.sequence)] ||
                sampled[static_cast<std::size_t>(sample.sequence)] || sample.token < 0 ||
                static_cast<std::size_t>(sample.token) >= runner->info().vocab_size) {
                throw std::runtime_error("model runner returned an invalid or duplicate sample");
            }
            sampled[static_cast<std::size_t>(sample.sequence)] = true;
        }
        ++counters.batches;
        counters.mixed_batches += static_cast<std::uint64_t>(plan.mixed());
        counters.prefill_tokens += plan.prefill_tokens;
        counters.decode_tokens += plan.decode_tokens;
        counters.max_batch_tokens = std::max(counters.max_batch_tokens, plan.token_count());
        counters.max_batch_sequences = std::max(counters.max_batch_sequences, plan.slices.size());
        std::vector<Active*> by_sequence(config.max_active, nullptr);
        for (const auto& slice : plan.slices) {
            auto& item = *active[slice.key];
            by_sequence[static_cast<std::size_t>(item.sequence)] = &item;
            item.last_scheduled = Clock::now();
            if (slice.prefill) {
                item.processed += slice.tokens;
                if (item.processed == item.request->prompt_.size() && !expired(item.request)) {
                    cache_prompt(item);
                }
            }
        }
        for (const auto& sample : samples) {
            auto& item = *by_sequence[static_cast<std::size_t>(sample.sequence)];
            SliceTelemetry* observed = nullptr;
            if (record) {
                for (std::size_t i = 0; i < record->sequences; ++i) {
                    if (record->slices[i].sequence == sample.sequence) {
                        observed = &record->slices[i];
                        observed->sampled_token = sample.token;
                        break;
                    }
                }
            }
            if (item.processed != item.request->prompt_.size()) {
                throw std::runtime_error("model runner sampled an unfinished prefill");
            }
            if (expired(item.request)) {
                continue;
            }
            item.last_token = sample.token;
            ++item.generated;
            ++counters.generated_tokens;
            if (!item.first_token) {
                item.first_token = Clock::now();
            }
            bool eog;
            std::string piece;
            {
                std::lock_guard lock(tokenizer_mutex);
                eog = runner->is_eog(sample.token) && !item.request->input_.ignore_eos;
                if (!eog) { piece = runner->token_piece(sample.token); }
            }
            Event event;
            event.token = sample.token;
            event.text = item.text.append(piece);
            event.usage = {item.request->prompt_.size(), item.generated, item.reused};
            if (capture.mode != TelemetryMode::off) {
                event.telemetry = TokenTelemetry{batch_id, item.request->order_, item.generated - 1,
                                                 elapsed_ns(Clock::now())};
                if (observed) { observed->emitted_ns = event.telemetry->engine_elapsed_ns; }
            }
            const auto emitted = item.request->emit(std::move(event));
            if (observed) { observed->emitted = emitted; }
            if (!emitted) {
                finish(item, "backpressure", 429, "client event buffer is full");
            } else if (eog || item.generated >= item.request->input_.max_tokens) {
                finish(item, eog ? "stop" : "length");
            }
        }
        erase_finished();
        if (record) { record->completed = true; }
    }

    void drain_incoming() {
        std::lock_guard lock(mutex);
        while (!incoming.empty()) {
            waiting.push_back(std::move(incoming.front()));
            incoming.pop_front();
        }
    }

    void terminate_all(const std::string& reason, int status, const std::string& message,
                       bool check_backend = true) {
        runner->synchronize();
        if (check_backend) { require_reusable(); }
        drain_incoming();
        for (auto& item : active) {
            finish(*item, reason, status, message, check_backend);
        }
        active.clear();
        for (const auto& request : waiting) {
            terminal(request, terminal_event(request, reason, status, message));
        }
        waiting.clear();
        while (const auto sequence = prefixes.least_recently_used()) {
            evict(*sequence, check_backend);
        }
        counters.ready = false;
        if (capture.mode != TelemetryMode::off) { capture.resources_final = runner->resources(); }
        publish();
    }

    void run() {
        try {
            require_reusable();
            publish();
            while (!stopping.load()) {
                {
                    std::unique_lock lock(mutex);
                    if (active.empty() && waiting.empty() && incoming.empty()) {
                        cv.wait(lock, [&] { return stopping.load() || !incoming.empty(); });
                    }
                }
                if (stopping.load()) {
                    break;
                }
                const auto start = capture.mode != TelemetryMode::off ? Clock::now() : Clock::time_point{};
                drain_incoming();
                expire_requests();
                admit_waiting();
                if (!active.empty()) {
                    const auto admitted = capture.mode != TelemetryMode::off ? Clock::now() : Clock::time_point{};
                    iteration(start, admitted);
                } else if (!waiting.empty()) {
                    throw std::logic_error("no request fits an otherwise idle KV pool");
                }
                publish();
            }
            terminate_all("cancelled", 503, "engine shutting down");
        } catch (const std::exception& error) {
            {
                std::lock_guard lock(mutex);
                failed = true;
            }
            counters.last_error = error.what();
            terminate_all("backend_error", 500, error.what(), false);
        }
    }
};

Engine::Engine(EngineConfig config, std::unique_ptr<ModelRunner> runner)
    : impl_(std::make_unique<Impl>(config, std::move(runner))) {}

Engine::~Engine() = default;

std::shared_ptr<RequestHandle> Engine::submit(RequestInput input) {
    const auto created = Clock::now();
    try {
        if (input.max_tokens == 0 || input.max_tokens >= impl_->config.max_model_len ||
            input.priority < 0 || input.priority > 3 || input.timeout_ms < 1 || input.timeout_ms > 300000 ||
            input.id.size() > 128 || input.cache_namespace.empty() || input.cache_namespace.size() > 64) {
            throw RequestError(422, "invalid_request", "invalid generation limits, priority, ID, or cache namespace");
        }
        std::vector<Token> prompt;
        if (const auto* text = std::get_if<std::string>(&input.prompt)) {
            if (text->size() > 262144) {
                throw RequestError(413, "prompt_too_large", "prompt exceeds 256 KiB");
            }
            prompt = tokenize(*text);
        } else {
            prompt = std::get<std::vector<Token>>(input.prompt);
        }
        if (prompt.empty() || prompt.size() > impl_->config.max_model_len - input.max_tokens) {
            throw RequestError(422, "context_length_exceeded", "require a nonempty prompt and prompt_tokens + max_tokens <= max_model_len");
        }
        for (const auto token : prompt) {
            if (token < 0 || static_cast<std::size_t>(token) >= impl_->runner->info().vocab_size) {
                throw RequestError(422, "invalid_token", "prompt contains an out-of-vocabulary token");
            }
        }
        std::lock_guard lock(impl_->mutex);
        if (impl_->stopping.load() || impl_->failed) {
            throw RequestError(503, "unavailable", "engine is not accepting requests");
        }
        if (impl_->registry.size() >= impl_->config.queue_capacity) {
            throw RequestError(429, "queue_full", "outstanding request limit reached");
        }
        const auto order = ++impl_->next_order;
        if (input.id.empty()) {
            input.id = "cmpl-" + std::to_string(order);
            while (impl_->registry.contains(input.id)) {
                input.id = "cmpl-" + std::to_string(++impl_->next_order);
            }
        } else if (impl_->registry.contains(input.id)) {
            throw RequestError(409, "duplicate_request_id", "request ID is already in flight");
        }
        auto request = std::shared_ptr<RequestHandle>(new RequestHandle(
            std::move(input), std::move(prompt), impl_->config.event_buffer_size, order, created));
        impl_->registry.emplace(request->id(), request);
        impl_->incoming.push_back(request);
        ++impl_->accepted;
        impl_->cv.notify_one();
        return request;
    } catch (...) {
        std::lock_guard lock(impl_->mutex);
        ++impl_->rejected;
        throw;
    }
}

bool Engine::cancel(const std::string& id) {
    std::lock_guard lock(impl_->mutex);
    const auto request = impl_->registry.find(id);
    if (request == impl_->registry.end()) {
        return false;
    }
    request->second->cancel();
    impl_->cv.notify_one();
    return true;
}

void Engine::stop() { impl_->stop(); }

const TelemetryCapture& Engine::telemetry() const {
    if (impl_->worker.joinable()) {
        throw std::logic_error("telemetry is available only after Engine::stop()");
    }
    return impl_->capture;
}

Statistics Engine::statistics() const {
    std::lock_guard lock(impl_->mutex);
    auto result = impl_->snapshot;
    result.ready = result.ready && !impl_->stopping.load() && !impl_->failed;
    result.accepted = impl_->accepted;
    result.rejected = impl_->rejected;
    result.outstanding_requests = impl_->registry.size();
    result.waiting_requests = result.outstanding_requests -
        std::min(result.outstanding_requests, result.active_requests);
    return result;
}

const EngineConfig& Engine::config() const noexcept { return impl_->config; }
const ModelInfo& Engine::model_info() const noexcept { return impl_->runner->info(); }
BackendCapabilities Engine::capabilities() const noexcept { return impl_->capabilities; }

std::vector<Token> Engine::tokenize(std::string_view text) const {
    std::lock_guard lock(impl_->tokenizer_mutex);
    return impl_->runner->tokenize(text);
}

} // namespace llmserve
