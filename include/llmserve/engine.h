#pragma once

#include "llmserve/config.h"
#include "llmserve/model_runner.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace llmserve {

using Clock = std::chrono::steady_clock;

class RequestError : public std::runtime_error {
public:
    RequestError(int status, std::string code, std::string message)
        : std::runtime_error(std::move(message)), status(status), code(std::move(code)) {}
    int status;
    std::string code;
};

struct RequestInput {
    std::string id;
    std::variant<std::string, std::vector<Token>> prompt;
    std::size_t max_tokens = 64;
    int priority = 0;
    int timeout_ms = 30000;
    bool ignore_eos = false;
    std::string cache_namespace = "default";
};

struct Usage {
    std::size_t prompt_tokens = 0;
    std::size_t completion_tokens = 0;
    std::size_t cached_tokens = 0;
};

struct Timings {
    double queue_ms = 0;
    std::optional<double> ttft_ms;
    double total_ms = 0;
};

struct Event {
    enum class Kind { token, done, error };
    Kind kind = Kind::token;
    std::string text;
    std::optional<Token> token;
    std::string finish_reason;
    std::string error_code;
    std::string error_message;
    int status = 200;
    Usage usage;
    Timings timings;
};

class RequestHandle {
public:
    const std::string& id() const noexcept { return input_.id; }
    std::optional<Event> next(std::chrono::milliseconds timeout);
    void cancel() noexcept { cancelled_.store(true); }
    bool finished() const;

private:
    friend class Engine;
    struct Access;
    RequestHandle(RequestInput input, std::vector<Token> prompt, std::size_t capacity,
                  std::uint64_t order, Clock::time_point created);
    bool emit(Event event);
    RequestInput input_;
    std::vector<Token> prompt_;
    std::size_t capacity_;
    std::uint64_t order_;
    Clock::time_point created_;
    Clock::time_point deadline_;
    std::atomic<bool> cancelled_{false};
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Event> events_;
    bool finished_ = false;
};

struct Statistics {
    bool ready = true;
    std::string last_error;
    std::uint64_t accepted = 0;
    std::uint64_t rejected = 0;
    std::uint64_t completed = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t timed_out = 0;
    std::uint64_t failed = 0;
    std::uint64_t batches = 0;
    std::uint64_t mixed_batches = 0;
    std::uint64_t prefill_tokens = 0;
    std::uint64_t decode_tokens = 0;
    std::uint64_t generated_tokens = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cached_tokens = 0;
    std::uint64_t cache_evictions = 0;
    std::size_t outstanding_requests = 0;
    std::size_t active_requests = 0;
    std::size_t waiting_requests = 0;
    std::size_t max_batch_tokens = 0;
    std::size_t max_batch_sequences = 0;
    std::size_t kv_total_blocks = 0;
    std::size_t kv_used_blocks = 0;
    std::size_t kv_active_unique_blocks = 0;
    std::size_t prefix_entries = 0;
    std::size_t prefix_tokens = 0;
};

class Engine {
public:
    Engine(EngineConfig config, std::unique_ptr<ModelRunner> runner);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    std::shared_ptr<RequestHandle> submit(RequestInput input);
    bool cancel(const std::string& id);
    void stop();
    Statistics statistics() const;
    const EngineConfig& config() const noexcept;
    const ModelInfo& model_info() const noexcept;
    std::vector<Token> tokenize(std::string_view text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llmserve
