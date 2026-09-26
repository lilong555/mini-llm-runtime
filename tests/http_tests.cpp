#include "test_support.h"
#include "../apps/options.h"

#include "httplib.h"
#include <nlohmann/json.hpp>

#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>

using json = nlohmann::json;
using namespace std::chrono_literals;

namespace {

json get(httplib::Client& client, const char* path) {
    const auto result = client.Get(path);
    CHECK(result && result->status == 200);
    return json::parse(result->body);
}

json idle(httplib::Client& client) {
    for (int i = 0; i < 200; ++i) {
        const auto stats = get(client, "/metrics");
        if (stats.at("outstanding_requests") == 0 && stats.at("active_requests") == 0) {
            CHECK(stats.at("kv_credits").at("active_unique_blocks") == 0);
            if (stats.at("backend") == "minillm-cuda") {
                CHECK(stats.at("resources").at("live_tokens") == 0);
                CHECK(stats.at("resources").at("state_valid") == true && stats.at("resources").at("reusable") == true);
                CHECK(stats.at("resources").at("resident_kv_payload_bytes").get<std::size_t>() > 0);
                CHECK(stats.at("resources").at("live_kv_pages").is_null());
            }
            return stats;
        }
        std::this_thread::sleep_for(50ms);
    }
    throw std::runtime_error("server did not drain after terminal request");
}

struct StreamResult {
    std::string text;
    std::vector<std::int32_t> tokens;
    json usage;
    std::string error;
    int done = 0;
    int terminal = 0;
    json telemetry = json::array();
};

class Frames {
public:
    template<class Callback>
    bool consume(const char* data, std::size_t size, Callback callback) {
        pending_.append(data, size);
        std::size_t position;
        while ((position = pending_.find("\n\n")) != std::string::npos) {
            auto frame = pending_.substr(0, position);
            pending_.erase(0, position + 2);
            if (frame.starts_with("data: ") && !callback(frame.substr(6))) {
                return false;
            }
        }
        return true;
    }
private:
    std::string pending_;
};

void capture(StreamResult& output, const std::string& payload) {
    if (payload == "[DONE]") {
        ++output.done;
        CHECK(output.done == 1 && output.terminal == 1);
        return;
    }
    CHECK(output.done == 0 && output.terminal == 0);
    const auto value = json::parse(payload);
    if (value.contains("error")) {
        ++output.terminal;
        output.error = value.at("error").at("code").get<std::string>();
    } else {
        if (!value.at("choices").at(0).at("finish_reason").is_null()) { ++output.terminal; }
        output.text += value.at("choices").at(0).at("text").get<std::string>();
        if (value.contains("token_id")) {
            output.tokens.push_back(value.at("token_id").get<std::int32_t>());
            output.telemetry.push_back(value.value("telemetry", json(nullptr)));
        }
        if (value.contains("usage")) {
            output.usage = value.at("usage");
        }
    }
}

StreamResult stream(httplib::Client& client, json body, const std::string& id = "") {
    body["stream"] = true;
    Frames frames;
    StreamResult output;
    const auto response = client.Post("/v1/completions", id.empty() ? httplib::Headers{} :
        httplib::Headers{{"X-Request-ID", id}}, body.dump(), "application/json",
        [&](const char* data, std::size_t size) {
            return frames.consume(data, size, [&](const std::string& payload) {
                capture(output, payload);
                return true;
            });
        });
    CHECK(response && response->status == 200 && output.done == 1 && output.terminal == 1);
    return output;
}

void save(const std::string& path, const json& report) {
    if (path.empty()) {
        return;
    }
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("cannot create HTTP verification report");
    }
    file << report.dump(2) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    json report{{"status", "failed"}, {"checks", json::array()}};
    std::string output_path;
    try {
        Options options(argc, argv, {"--port", "--output", "--shutdown-file"});
        if (options.has("--help")) {
            std::cout << "llmserve-http-tests [--port 8000] [--output REPORT.json] [--shutdown-file OWN_MARKER]\n";
            return 0;
        }
        output_path = options.get("--output");
        const auto port = static_cast<int>(options.integer("--port", 8000, 1024, 65535));
        httplib::Client client("127.0.0.1", port);
        client.set_read_timeout(180);
        auto& checks = report["checks"];
        CHECK(get(client, "/health").at("ready") == true);
        const auto before = get(client, "/metrics");
        report["server_before"] = before;
        const auto models = get(client, "/v1/models");
        CHECK(models.at("data").at(0).at("id") == before.at("model"));
        checks.push_back("health_and_model_identity");
        const auto& capabilities = before.at("capabilities");
        CHECK(capabilities.at("synchronous_execute") == true);
        CHECK(capabilities.at("max_sequences") >= before.at("max_active"));
        if (before.at("backend") == "minillm-cuda") {
            CHECK(before.at("gpu") == true && before.at("gpu_layers") == 0 && before.at("kernel_mode") == "cuda-f32");
            CHECK(capabilities.at("prefix_copy") == false && capabilities.at("runtime_stage_profile") == false);
            CHECK(before.at("prefix_cache_entries") == 0 && before.at("prefix_cache_tokens") == 0);
            const auto& resources = before.at("resources");
            CHECK(resources.at("layout") == "contiguous" && resources.at("live_kv_pages").is_null());
            CHECK(resources.at("capacity_tokens").get<std::size_t>() ==
                  before.at("max_active").get<std::size_t>() * before.at("max_model_len").get<std::size_t>());
            CHECK(resources.at("snapshot_boundary") == "model_thread_publish");
            CHECK(resources.at("owned_device_bytes") >= resources.at("resident_kv_payload_bytes"));
            CHECK(before.at("initialization").at("weight_decode_upload_ns").get<std::uint64_t>() > 0);
            CHECK(before.at("initialization").at("weight_decode_upload_ns") <=
                  before.at("initialization").at("storage_initialization_ns"));
        } else if (before.at("backend") == "minillm") {
            CHECK(capabilities.at("prefix_copy") == true && capabilities.at("runtime_stage_profile") == true);
            CHECK(before.at("resources").at("layout") == "paged");
            CHECK(before.at("resources").at("live_tokens").is_null());
        } else {
            CHECK(before.at("backend") == "llama.cpp" && before.at("resources").is_null());
        }
        checks.push_back("backend_capabilities_and_resource_semantics");

        const json body{{"prompt", "The capital of France is"}, {"max_tokens", 8}, {"ignore_eos", true}};
        auto response = client.Post("/v1/completions", body.dump(), "application/json");
        CHECK(response && response->status == 200);
        const auto full = json::parse(response->body);
        const auto streamed = stream(client, body);
        CHECK(streamed.error.empty());
        CHECK(streamed.text == full.at("choices").at(0).at("text").get<std::string>());
        CHECK(streamed.tokens == full.at("token_ids").get<std::vector<std::int32_t>>());
        CHECK(streamed.usage.at("completion_tokens") == 8);
        CHECK(streamed.usage.at("prompt_tokens") == full.at("usage").at("prompt_tokens"));
        for (std::size_t i = 0; i < streamed.telemetry.size(); ++i) {
            const auto& token = streamed.telemetry[i];
            if (before.value("telemetry_mode", "off") == "off") {
                CHECK(token.is_null());
            } else {
                CHECK(token.at("token_index") == i && token.at("batch_id").get<std::uint64_t>() > 0);
                CHECK(token.at("engine_elapsed_ns").get<std::uint64_t>() > 0);
                if (i > 0) {
                    CHECK(token.at("batch_id") > streamed.telemetry[i - 1].at("batch_id"));
                    CHECK(token.at("request_order") == streamed.telemetry[0].at("request_order"));
                }
            }
        }
        checks.push_back("stream_nonstream_text_tokens_and_usage_match");
        report["sample"] = full;

        const json unicode_body{{"prompt", "你好，我是"}, {"max_tokens", 8}, {"ignore_eos", true}};
        const auto unicode = client.Post("/v1/completions", unicode_body.dump(), "application/json");
        CHECK(unicode && unicode->status == 200);
        const auto unicode_full = json::parse(unicode->body);
        const auto unicode_stream = stream(client, unicode_body);
        CHECK(unicode_stream.error.empty());
        CHECK(unicode_stream.tokens == unicode_full.at("token_ids").get<std::vector<std::int32_t>>());
        CHECK(unicode_stream.text == unicode_full.at("choices").at(0).at("text").get<std::string>());
        CHECK(std::any_of(unicode_stream.text.begin(), unicode_stream.text.end(),
                          [](unsigned char c) { return c >= 0x80; }));
        checks.push_back("utf8_stream_nonstream_match");

        for (const json invalid : {
            json{{"prompt", "x"}, {"temperature", 1}},
            json{{"prompt", "x"}, {"max_tokens", true}},
            json{{"prompt", "x"}, {"stream", "true"}},
            json{{"prompt", "x"}, {"top_p", 0.9}},
            json{{"prompt", json::array({-1})}},
            json{{"prompt", "x"}, {"max_tokens", 0}, {"stream", true}}
        }) {
            const auto bad = client.Post("/v1/completions", invalid.dump(), "application/json");
            CHECK(bad && bad->status == 422);
        }
        const auto overflow = client.Post("/v1/completions",
            json{{"prompt", std::vector<int>(before.at("max_model_len").get<std::size_t>(), 1)},
                 {"max_tokens", 1}}.dump(), "application/json");
        CHECK(overflow && overflow->status == 422);
        const auto malformed = client.Post("/v1/completions", "{", "application/json");
        CHECK(malformed && malformed->status == 400);
        checks.push_back("strict_input_and_context_validation");

        std::vector<std::future<bool>> concurrent;
        for (int i = 0; i < 8; ++i) {
            concurrent.push_back(std::async(std::launch::async, [=] {
                httplib::Client other("127.0.0.1", port);
                other.set_read_timeout(180);
                const auto result = other.Post("/v1/completions",
                    json{{"prompt", "Request " + std::to_string(i) + ": explain memory."},
                        {"max_tokens", 4}, {"ignore_eos", true}}.dump(), "application/json");
                return result && result->status == 200 &&
                    json::parse(result->body).at("usage").at("completion_tokens") == 4;
            }));
        }
        for (auto& task : concurrent) {
            CHECK(task.get());
        }
        checks.push_back("eight_concurrent_requests");

        Frames cancel_frames;
        StreamResult cancelled;
        bool cancel_sent = false;
        const json long_stream{{"prompt", "List the numbers:"}, {"max_tokens", 512},
                                {"ignore_eos", true}, {"stream", true}, {"timeout_ms", 120000}};
        response = client.Post("/v1/completions", {{"X-Request-ID", "http-explicit-cancel"}},
            long_stream.dump(), "application/json", [&](const char* data, std::size_t size) {
                return cancel_frames.consume(data, size, [&](const std::string& payload) {
                    capture(cancelled, payload);
                    if (!cancel_sent && !cancelled.tokens.empty()) {
                        httplib::Client control("127.0.0.1", port);
                        const auto result = control.Delete("/v1/requests/http-explicit-cancel");
                        CHECK(result && result->status == 200);
                        cancel_sent = true;
                    }
                    return true;
                });
            });
        CHECK(response && response->status == 200);
        CHECK(cancel_sent && cancelled.error == "cancelled" && cancelled.done == 1 && cancelled.terminal == 1);
        idle(client);
        checks.push_back("explicit_live_stream_cancellation");

        const auto cancel_count = get(client, "/metrics").at("cancelled").get<std::uint64_t>();
        bool received_token = false;
        Frames disconnect_frames;
        httplib::Client disconnect("127.0.0.1", port);
        disconnect.set_read_timeout(180);
        disconnect.Post("/v1/completions", {{"X-Request-ID", "http-stream-disconnect"}},
            long_stream.dump(), "application/json", [&](const char* data, std::size_t size) {
                return disconnect_frames.consume(data, size, [&](const std::string& payload) {
                    if (payload != "[DONE]" && json::parse(payload).contains("token_id")) {
                        received_token = true;
                        return false;
                    }
                    return true;
                });
            });
        CHECK(received_token);
        CHECK(idle(client).at("cancelled").get<std::uint64_t>() > cancel_count);
        checks.push_back("stream_disconnect_reclaims_kv");

        httplib::Client nonstream("127.0.0.1", port);
        nonstream.set_read_timeout(180);
        auto long_body = long_stream;
        long_body["stream"] = false;
        auto nonstream_request = std::async(std::launch::async, [&] {
            return nonstream.Post("/v1/completions", {{"X-Request-ID", "http-nonstream-disconnect"}},
                                  long_body.dump(), "application/json");
        });
        bool observed = false;
        for (int i = 0; i < 200; ++i) {
            if (get(client, "/metrics").at("outstanding_requests").get<std::size_t>() > 0) {
                observed = true;
                break;
            }
            std::this_thread::sleep_for(5ms);
        }
        nonstream.stop();
        nonstream_request.get();
        CHECK(observed);
        idle(client);
        checks.push_back("nonstream_disconnect_reclaims_kv");

        const auto before_slow = idle(client);
        httplib::Client slow("127.0.0.1", port);
        slow.set_read_timeout(180);
        slow.set_socket_options([](socket_t socket) {
            CHECK(httplib::set_socket_opt(socket, SOL_SOCKET, SO_RCVBUF, 1024));
        });
        Frames slow_frames;
        bool slow_received = false;
        slow.Post("/v1/completions", {{"X-Request-ID", "http-slow-consumer"}},
            long_stream.dump(), "application/json", [&](const char* data, std::size_t size) {
                return slow_frames.consume(data, size, [&](const std::string& payload) {
                    if (payload != "[DONE]" && json::parse(payload).contains("token_id")) {
                        slow_received = true;
                        std::this_thread::sleep_for(350ms);
                        httplib::Client control("127.0.0.1", port);
                        const auto snapshot = get(control, "/metrics");
                        CHECK(snapshot.at("outstanding_requests") <= snapshot.at("queue_capacity"));
                        CHECK(snapshot.at("active_requests") <= snapshot.at("max_active"));
                        CHECK(snapshot.at("kv_credits").at("reserved_unique_blocks") <= snapshot.at("kv_credits").at("total_blocks"));
                        if (snapshot.at("backend") == "minillm-cuda") {
                            CHECK(snapshot.at("resources").at("live_tokens") <= snapshot.at("resources").at("capacity_tokens"));
                            CHECK(snapshot.at("resources").at("owned_device_bytes") == before.at("resources").at("owned_device_bytes"));
                        }
                        return false;
                    }
                    return true;
                });
            });
        CHECK(slow_received);
        const auto after_slow = idle(client);
        const auto slow_failures = after_slow.at("failed").get<std::uint64_t>() - before_slow.at("failed").get<std::uint64_t>();
        CHECK(slow_failures <= 1 && after_slow.at("ready") == true && after_slow.at("last_error") == "");
        CHECK(after_slow.at("completed").get<std::uint64_t>() + after_slow.at("cancelled").get<std::uint64_t>() +
              after_slow.at("failed").get<std::uint64_t>() ==
              before_slow.at("completed").get<std::uint64_t>() + before_slow.at("cancelled").get<std::uint64_t>() +
              before_slow.at("failed").get<std::uint64_t>() + 1);
        checks.push_back("slow_socket_bounded_and_reclaimed");

        auto deadline = body;
        deadline["timeout_ms"] = 1;
        deadline["max_tokens"] = 64;
        const auto timed_out = client.Post("/v1/completions", deadline.dump(), "application/json");
        CHECK(timed_out && timed_out->status == 408);
        CHECK(json::parse(timed_out->body).at("error").at("code") == "timeout");
        checks.push_back("deadline_returns_408");
        const auto after = idle(client);
        CHECK(after.at("failed").get<std::uint64_t>() == before.at("failed").get<std::uint64_t>() + slow_failures);
        report["server_after"] = after;
        const auto shutdown_file = options.get("--shutdown-file");
        if (!shutdown_file.empty()) {
            CHECK(!std::filesystem::exists(shutdown_file));
            const auto count = before.at("max_active").get<std::size_t>() + 2;
            CHECK(count <= before.at("queue_capacity").get<std::size_t>());
            std::vector<std::future<StreamResult>> stopping;
            for (std::size_t i = 0; i < count; ++i) {
                stopping.push_back(std::async(std::launch::async, [&, i] {
                    httplib::Client other("127.0.0.1", port);
                    other.set_read_timeout(180);
                    return stream(other, long_stream, "http-stop-" + std::to_string(i));
                }));
            }
            json at_stop;
            for (int i = 0; i < 400; ++i) {
                at_stop = get(client, "/metrics");
                if (at_stop.at("outstanding_requests") == count &&
                    at_stop.at("active_requests").get<std::size_t>() > 0 &&
                    at_stop.at("waiting_requests").get<std::size_t>() > 0) { break; }
                std::this_thread::sleep_for(5ms);
            }
            const bool active_and_queued = at_stop.at("outstanding_requests") == count &&
                at_stop.at("active_requests").get<std::size_t>() > 0 &&
                at_stop.at("waiting_requests").get<std::size_t>() > 0;
            std::ofstream marker(shutdown_file);
            marker.close();
            CHECK(marker && active_and_queued);
            for (auto& task : stopping) {
                const auto stopped = task.get();
                CHECK(stopped.error == "cancelled" && stopped.terminal == 1 && stopped.done == 1);
            }
            report["shutdown"] = {{"requests", count}, {"active", at_stop.at("active_requests")},
                                   {"queued", at_stop.at("waiting_requests")}, {"single_terminals", count}};
            checks.push_back("shutdown_active_and_queued_single_terminals");
        }
        report["passed"] = checks.size();
        report["status"] = "passed";
        save(output_path, report);
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        report["error"] = error.what();
        save(output_path, report);
        std::cerr << report.dump(2) << '\n';
        return 1;
    }
}
