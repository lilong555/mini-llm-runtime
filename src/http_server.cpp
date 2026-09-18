#include "llmserve/http_server.h"

#include "llmserve/engine.h"
#include "httplib.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <regex>
#include <set>
#include <thread>

namespace llmserve {
namespace {

using json = nlohmann::json;
using namespace std::chrono_literals;
volatile std::sig_atomic_t interrupted = 0;

void signal_handler(int) { interrupted = 1; }

json usage_json(const Usage& usage) {
    return {{"prompt_tokens", usage.prompt_tokens}, {"completion_tokens", usage.completion_tokens},
            {"total_tokens", usage.prompt_tokens + usage.completion_tokens},
            {"prompt_tokens_details", {{"cached_tokens", usage.cached_tokens}}}};
}

json timings_json(const Timings& timings) {
    return {{"queue_ms", timings.queue_ms},
            {"ttft_ms", timings.ttft_ms ? json(*timings.ttft_ms) : json(nullptr)},
            {"total_ms", timings.total_ms}};
}

json error_json(const std::string& code, const std::string& message) {
    return {{"error", {{"code", code}, {"message", message}, {"type", "runtime_error"}}}};
}

void respond(httplib::Response& response, int status, const json& value) {
    response.status = status;
    response.set_content(value.dump(), "application/json");
}

std::int64_t integer(const json& body, const char* key, std::int64_t fallback,
                     std::int64_t low, std::int64_t high) {
    if (!body.contains(key)) {
        return fallback;
    }
    const auto& value = body.at(key);
    if (!value.is_number_integer() || (value.is_number_unsigned() &&
        value.get<std::uint64_t>() > static_cast<std::uint64_t>(high))) {
        throw RequestError(422, "invalid_request", std::string(key) + " must be an integer in range");
    }
    const auto result = value.get<std::int64_t>();
    if (result < low || result > high) {
        throw RequestError(422, "invalid_request", std::string(key) + " is out of range");
    }
    return result;
}

json parse_body(const httplib::Request& request) {
    auto value = json::parse(request.body);
    if (!value.is_object()) {
        throw RequestError(422, "invalid_request", "request body must be a JSON object");
    }
    return value;
}

RequestInput parse_request(const json& body, const httplib::Request& http, const Engine& engine) {
    static const std::set<std::string> allowed{
        "prompt", "model", "max_tokens", "temperature", "stream", "priority",
        "timeout_ms", "ignore_eos", "cache_namespace", "n", "stop"};
    for (const auto& [key, value] : body.items()) {
        (void)value;
        if (!allowed.contains(key)) {
            throw RequestError(422, "unsupported_parameter", "unsupported parameter: " + key);
        }
    }
    RequestInput input;
    if (!body.contains("prompt")) {
        throw RequestError(422, "invalid_request", "prompt is required");
    }
    if (body.at("prompt").is_string()) {
        input.prompt = body.at("prompt").get<std::string>();
    } else if (body.at("prompt").is_array()) {
        const auto& prompt = body.at("prompt");
        if (prompt.size() > engine.config().max_model_len) {
            throw RequestError(422, "context_length_exceeded", "prompt token count exceeds max_model_len");
        }
        std::vector<Token> tokens;
        for (const auto& token : prompt) {
            if (!token.is_number_integer() || token < 0 || token >= engine.model_info().vocab_size) {
                throw RequestError(422, "invalid_token", "prompt token IDs must be integers in the vocabulary");
            }
            tokens.push_back(token.get<Token>());
        }
        input.prompt = std::move(tokens);
    } else {
        throw RequestError(422, "invalid_request", "prompt must be a string or an array of token IDs");
    }
    if (body.contains("model") &&
        (!body["model"].is_string() || body["model"].get<std::string>() != engine.model_info().model)) {
        throw RequestError(404, "model_not_found", "the requested model is not loaded");
    }
    if (body.contains("temperature") &&
        (!body["temperature"].is_number() || body["temperature"].get<double>() != 0.0)) {
        throw RequestError(422, "unsupported_parameter", "only greedy decoding (temperature=0) is supported");
    }
    integer(body, "n", 1, 1, 1);
    if (body.contains("stop") && !body["stop"].is_null() &&
        !(body["stop"].is_array() && body["stop"].empty())) {
        throw RequestError(422, "unsupported_parameter", "custom stop sequences are not supported");
    }
    for (const auto* field : {"stream", "ignore_eos"}) {
        if (body.contains(field) && !body[field].is_boolean()) {
            throw RequestError(422, "invalid_request", std::string(field) + " must be boolean");
        }
    }
    input.max_tokens = static_cast<std::size_t>(integer(body, "max_tokens", 64, 1,
        static_cast<std::int64_t>(engine.config().max_model_len - 1)));
    input.priority = static_cast<int>(integer(body, "priority", 0, 0, 3));
    input.timeout_ms = static_cast<int>(integer(body, "timeout_ms", 30000, 1, 300000));
    input.ignore_eos = body.value("ignore_eos", false);
    if (body.contains("cache_namespace")) {
        if (!body["cache_namespace"].is_string()) {
            throw RequestError(422, "invalid_request", "cache_namespace must be a string");
        }
        input.cache_namespace = body["cache_namespace"].get<std::string>();
    }
    input.id = http.get_header_value("X-Request-ID");
    if (!input.id.empty() && !std::regex_match(input.id, std::regex("[A-Za-z0-9_.:-]{1,128}"))) {
        throw RequestError(422, "invalid_request", "X-Request-ID contains invalid characters");
    }
    return input;
}

json completion_json(const std::string& id, const std::string& model,
                     std::int64_t created, const Event& event) {
    if (event.kind == Event::Kind::error) {
        auto result = error_json(event.error_code, event.error_message);
        result["id"] = id;
        result["usage"] = usage_json(event.usage);
        result["timings"] = timings_json(event.timings);
        return result;
    }
    json result{{"id", id}, {"object", "text_completion"}, {"created", created}, {"model", model},
        {"choices", json::array({{{"index", 0}, {"text", event.text}, {"logprobs", nullptr},
            {"finish_reason", event.kind == Event::Kind::done ? json(event.finish_reason) : json(nullptr)}}})}};
    if (event.token) {
        result["token_id"] = *event.token;
    }
    if (event.kind == Event::Kind::done) {
        result["usage"] = usage_json(event.usage);
        result["timings"] = timings_json(event.timings);
    }
    return result;
}

json metrics_json(const Engine& engine) {
    const auto s = engine.statistics();
    const auto& c = engine.config();
    const auto& m = engine.model_info();
    return {
        {"ready", s.ready}, {"backend", m.backend}, {"model", m.model}, {"device", m.device},
        {"gpu", m.gpu}, {"policy", policy_name(c.policy)},
        {"llama_commit", "911f6cdc8ab8a530b2bee09ee61471a6f3178eeb"},
        {"context_tokens", c.context_tokens}, {"max_model_len", c.max_model_len},
        {"batch_tokens", c.batch_tokens}, {"prefill_chunk", c.prefill_chunk},
        {"max_active", c.max_active}, {"queue_capacity", c.queue_capacity},
        {"prefix_cache_enabled", c.prefix_cache_entries > 0},
        {"accepted", s.accepted}, {"rejected", s.rejected}, {"completed", s.completed},
        {"cancelled", s.cancelled}, {"timed_out", s.timed_out}, {"failed", s.failed},
        {"batches", s.batches}, {"mixed_batches", s.mixed_batches},
        {"prefill_tokens", s.prefill_tokens}, {"decode_tokens", s.decode_tokens},
        {"generated_tokens", s.generated_tokens}, {"cache_hits", s.cache_hits},
        {"cached_tokens", s.cached_tokens}, {"cache_evictions", s.cache_evictions},
        {"max_batch_tokens", s.max_batch_tokens}, {"max_batch_sequences", s.max_batch_sequences},
        {"outstanding_requests", s.outstanding_requests}, {"active_requests", s.active_requests},
        {"waiting_requests", s.waiting_requests},
        {"kv_credits", {{"block_size", c.block_size}, {"total_blocks", s.kv_total_blocks},
             {"reserved_unique_blocks", s.kv_used_blocks}, {"active_unique_blocks", s.kv_active_unique_blocks}}},
        {"prefix_cache", {{"entries", s.prefix_entries}, {"tokens", s.prefix_tokens}}},
        {"last_error", s.last_error}
    };
}

} // namespace

bool serve_http(Engine& engine, int port, const std::string& shutdown_file) {
    httplib::Server server;
    const auto threads = std::min<std::size_t>(256, engine.config().queue_capacity + 16);
    server.new_task_queue = [threads] { return new httplib::ThreadPool(8, threads, 64); };
    server.set_payload_max_length(1048576);
    server.set_read_timeout(10);
    server.set_write_timeout(5);
    server.set_keep_alive_timeout(5);
    server.set_exception_handler([](const auto&, auto& response, std::exception_ptr error) {
        try {
            std::rethrow_exception(error);
        } catch (const RequestError& value) {
            respond(response, value.status, error_json(value.code, value.what()));
            if (value.status == 429) {
                response.set_header("Retry-After", "1");
            }
        } catch (const json::parse_error&) {
            respond(response, 400, error_json("invalid_json", "malformed JSON body"));
        } catch (const json::exception&) {
            respond(response, 422, error_json("invalid_request", "invalid JSON field type"));
        } catch (const std::exception& value) {
            respond(response, 500, error_json("internal_error", value.what()));
        }
    });
    server.Get("/health", [&](const auto&, auto& response) {
        const auto ready = engine.statistics().ready;
        respond(response, ready ? 200 : 503, {{"ready", ready}, {"backend", engine.model_info().backend}});
    });
    server.Get("/", [&](const auto&, auto& response) {
        respond(response, 200, {{"name", "MiniLLM / LLMServe"}, {"backend", engine.model_info().backend},
            {"model", engine.model_info().model}, {"device", engine.model_info().device}});
    });
    server.Get("/metrics", [&](const auto&, auto& response) { respond(response, 200, metrics_json(engine)); });
    server.Get("/v1/models", [&](const auto&, auto& response) {
        respond(response, 200, {{"object", "list"},
            {"data", json::array({{{"id", engine.model_info().model}, {"object", "model"}, {"owned_by", "local"}}})}});
    });
    server.Post("/tokenize", [&](const auto& request, auto& response) {
        const auto body = parse_body(request);
        if (!body.contains("text") || !body["text"].is_string()) {
            throw RequestError(422, "invalid_request", "text must be a string");
        }
        const auto text = body["text"].template get<std::string>();
        if (text.size() > 262144) {
            throw RequestError(413, "prompt_too_large", "text exceeds 256 KiB");
        }
        const auto tokens = engine.tokenize(text);
        respond(response, 200, {{"tokens", tokens}, {"count", tokens.size()}});
    });
    server.Delete(R"(/v1/requests/([A-Za-z0-9_.:-]+))", [&](const auto& request, auto& response) {
        const auto id = request.matches[1].str();
        if (engine.cancel(id)) {
            respond(response, 200, {{"id", id}, {"cancelled", true}});
        } else {
            respond(response, 404, error_json("request_not_found", "request is not in flight"));
        }
    });
    server.Post("/v1/completions", [&](const auto& request, auto& response) {
        const auto body = parse_body(request);
        auto handle = engine.submit(parse_request(body, request, engine));
        response.set_header("X-Request-ID", handle->id());
        const auto model = engine.model_info().model;
        const auto created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (body.value("stream", false)) {
            response.set_header("Cache-Control", "no-cache");
            response.set_header("X-Accel-Buffering", "no");
            response.set_chunked_content_provider("text/event-stream",
                [handle, model, created, closed = request.is_connection_closed]
                (std::size_t, httplib::DataSink& sink) {
                    while (true) {
                        if (closed() || !sink.is_writable()) {
                            handle->cancel();
                            return false;
                        }
                        const auto event = handle->next(100ms);
                        if (!event) {
                            continue;
                        }
                        const auto data = "data: " + completion_json(handle->id(), model, created, *event).dump() + "\n\n";
                        if (!sink.write(data.data(), data.size())) {
                            handle->cancel();
                            return false;
                        }
                        if (event->kind != Event::Kind::token) {
                            constexpr std::string_view done = "data: [DONE]\n\n";
                            if (!sink.write(done.data(), done.size())) {
                                return false;
                            }
                            sink.done();
                            return true;
                        }
                        return true;
                    }
                },
                [handle](bool) {
                    if (!handle->finished()) {
                        handle->cancel();
                    }
                });
            return;
        }
        std::string text;
        std::vector<Token> tokens;
        while (true) {
            if (request.is_connection_closed()) {
                handle->cancel();
                response.status = 499;
                return;
            }
            const auto event = handle->next(50ms);
            if (!event) {
                continue;
            }
            text += event->text;
            if (event->token) {
                tokens.push_back(*event->token);
            }
            if (event->kind != Event::Kind::token) {
                auto terminal = *event;
                terminal.text = std::move(text);
                auto result = completion_json(handle->id(), model, created, terminal);
                result["token_ids"] = tokens;
                respond(response, event->status, result);
                return;
            }
        }
    });
    if (!shutdown_file.empty() && std::filesystem::exists(shutdown_file)) {
        throw std::runtime_error("shutdown marker already exists");
    }
    interrupted = 0;
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::jthread monitor([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            std::error_code error;
            if (interrupted || (!shutdown_file.empty() && std::filesystem::exists(shutdown_file, error))) {
                engine.stop();
                server.stop();
                return;
            }
            std::this_thread::sleep_for(50ms);
        }
    });
    std::cout << "LLMServe " << engine.model_info().backend << " http://127.0.0.1:" << port << '\n' << std::flush;
    const auto success = server.listen("127.0.0.1", port);
    monitor.request_stop();
    monitor.join();
    engine.stop();
    return success;
}

} // namespace llmserve
