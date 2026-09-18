#include "test_support.h"
#include "../apps/options.h"

#include "httplib.h"
#include <nlohmann/json.hpp>

#include <chrono>
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
        return;
    }
    const auto value = json::parse(payload);
    if (value.contains("error")) {
        output.error = value.at("error").at("code").get<std::string>();
    } else {
        output.text += value.at("choices").at(0).at("text").get<std::string>();
        if (value.contains("token_id")) {
            output.tokens.push_back(value.at("token_id").get<std::int32_t>());
        }
        if (value.contains("usage")) {
            output.usage = value.at("usage");
        }
    }
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
        Options options(argc, argv, {"--port", "--output"});
        if (options.has("--help")) {
            std::cout << "llmserve-http-tests [--port 8000] [--output REPORT.json]\n";
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

        const json body{{"prompt", "The capital of France is"}, {"max_tokens", 8}, {"ignore_eos", true}};
        auto response = client.Post("/v1/completions", body.dump(), "application/json");
        CHECK(response && response->status == 200);
        const auto full = json::parse(response->body);
        Frames frames;
        StreamResult streamed;
        auto streaming = body;
        streaming["stream"] = true;
        response = client.Post("/v1/completions", {}, streaming.dump(), "application/json",
            [&](const char* data, std::size_t size) {
                return frames.consume(data, size, [&](const std::string& payload) {
                    capture(streamed, payload);
                    return true;
                });
            });
        CHECK(response && response->status == 200);
        CHECK(streamed.done == 1 && streamed.error.empty());
        CHECK(streamed.text == full.at("choices").at(0).at("text").get<std::string>());
        CHECK(streamed.tokens == full.at("token_ids").get<std::vector<std::int32_t>>());
        CHECK(streamed.usage.at("completion_tokens") == 8);
        CHECK(streamed.usage.at("prompt_tokens") == full.at("usage").at("prompt_tokens"));
        checks.push_back("stream_nonstream_text_tokens_and_usage_match");
        report["sample"] = full;

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
        CHECK(cancel_sent && cancelled.error == "cancelled" && cancelled.done == 1);
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

        auto deadline = body;
        deadline["timeout_ms"] = 1;
        deadline["max_tokens"] = 64;
        const auto timed_out = client.Post("/v1/completions", deadline.dump(), "application/json");
        CHECK(timed_out && timed_out->status == 408);
        CHECK(json::parse(timed_out->body).at("error").at("code") == "timeout");
        checks.push_back("deadline_returns_408");
        const auto after = idle(client);
        CHECK(after.at("failed") == before.at("failed"));
        report["server_after"] = after;
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
