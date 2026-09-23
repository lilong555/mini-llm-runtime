#include "options.h"

#include "httplib.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

json get_json(httplib::Client& client, const char* path) {
    const auto response = client.Get(path);
    if (!response || response->status != 200) {
        throw std::runtime_error(std::string("cannot query server: ") + path);
    }
    return json::parse(response->body);
}

std::ofstream output_file(const std::string& path) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("cannot create output: " + path);
    }
    return file;
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    const auto rank = q * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(rank);
    const auto upper = std::min(lower + 1, values.size() - 1);
    return values[lower] + (values[upper] - values[lower]) * (rank - static_cast<double>(lower));
}

struct Measurement {
    std::string id;
    std::string class_name;
    bool success = false;
    bool within_slo = false;
    std::string error;
    std::string finish_reason;
    int http_status = 0;
    int done_count = 0;
    int terminal_events = 0;
    double dispatch_lag_ms = 0;
    double ttft_ms = 0;
    double tpot_ms = 0;
    double e2e_ms = 0;
    double finished_s = 0;
    std::vector<double> token_times_ms;
    std::vector<std::int32_t> token_ids;
    json token_telemetry = json::array();
    json usage;
    json server_timings;
};

void replay_one(const json& row, const std::string& model, int port,
                Clock::time_point start, Measurement& measurement) {
    measurement.id = row.at("request_id").get<std::string>();
    measurement.class_name = row.value("class_name", "default");
    const auto arrival = row.at("arrival_s").get<double>();
    const auto scheduled = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(arrival));
    std::this_thread::sleep_until(scheduled);
    const auto sent = Clock::now();
    measurement.dispatch_lag_ms = std::chrono::duration<double, std::milli>(sent - scheduled).count();
    try {
        httplib::Client client("127.0.0.1", port);
        client.set_connection_timeout(10);
        client.set_read_timeout(300);
        auto body = row.at("request");
        body["model"] = model;
        body["stream"] = true;
        std::string pending;
        const auto response = client.Post("/v1/completions", {{"X-Request-ID", measurement.id}},
            body.dump(), "application/json", [&](const char* data, std::size_t length) {
                try {
                    pending.append(data, length);
                    if (pending.size() > 1048576) {
                        throw std::runtime_error("oversized SSE frame");
                    }
                    std::size_t boundary;
                    while ((boundary = pending.find("\n\n")) != std::string::npos) {
                        const auto frame = pending.substr(0, boundary);
                        pending.erase(0, boundary + 2);
                        if (!frame.starts_with("data: ")) {
                            continue;
                        }
                        if (measurement.done_count != 0) {
                            throw std::runtime_error("SSE data after DONE");
                        }
                        const auto payload = frame.substr(6);
                        if (payload == "[DONE]") {
                            ++measurement.done_count;
                            continue;
                        }
                        const auto event = json::parse(payload);
                        if (measurement.terminal_events != 0) {
                            throw std::runtime_error("SSE data after terminal event");
                        }
                        if (event.contains("error")) {
                            measurement.error = event["error"].value("code", "server_error");
                        }
                        if (event.contains("token_id")) {
                            measurement.token_ids.push_back(event["token_id"].get<std::int32_t>());
                            measurement.token_times_ms.push_back(
                                std::chrono::duration<double, std::milli>(Clock::now() - sent).count());
                            measurement.token_telemetry.push_back(event.value("telemetry", json(nullptr)));
                        }
                        if (event.contains("usage")) {
                            measurement.usage = event["usage"];
                            measurement.server_timings = event.value("timings", json::object());
                            ++measurement.terminal_events;
                            if (!event.contains("error")) {
                                measurement.finish_reason = event.at("choices").at(0).at("finish_reason").get<std::string>();
                            }
                        }
                    }
                    return true;
                } catch (const std::exception& error) {
                    measurement.error = error.what();
                    return false;
                }
            });
        measurement.http_status = response ? response->status : 0;
        if (response && response->status != 200 && measurement.error.empty() && !pending.empty()) {
            const auto error = json::parse(pending, nullptr, false);
            if (!error.is_discarded() && error.contains("error")) {
                measurement.error = error["error"].value("code", "server_error");
            }
        }
        if (!response && measurement.error.empty()) {
            measurement.error = httplib::to_string(response.error());
        }
        const auto maximum = body.value("max_tokens", std::size_t{64});
        const auto complete_length = measurement.finish_reason == "length" &&
            measurement.token_ids.size() == maximum;
        const auto complete_stop = measurement.finish_reason == "stop" &&
            !body.value("ignore_eos", false) && measurement.token_ids.size() <= maximum;
        measurement.success = response && response->status == 200 && measurement.done_count == 1 &&
            measurement.terminal_events == 1 && pending.empty() && (complete_length || complete_stop) &&
            measurement.error.empty() && !measurement.token_ids.empty() &&
            measurement.usage.value("completion_tokens", std::size_t{0}) == measurement.token_ids.size();
        if (!measurement.success && measurement.error.empty()) {
            measurement.error = "incomplete_or_rejected_response";
        }
        if (!measurement.token_times_ms.empty()) {
            measurement.ttft_ms = measurement.token_times_ms.front();
            if (measurement.token_times_ms.size() > 1) {
                measurement.tpot_ms = (measurement.token_times_ms.back() - measurement.ttft_ms) /
                    static_cast<double>(measurement.token_times_ms.size() - 1);
            }
        }
        measurement.within_slo = measurement.success &&
            measurement.ttft_ms <= row.value("ttft_slo_ms", 1500.0) &&
            measurement.tpot_ms <= row.value("tpot_slo_ms", 100.0);
    } catch (const std::exception& error) {
        measurement.error = error.what();
    }
    const auto finished = Clock::now();
    measurement.e2e_ms = std::chrono::duration<double, std::milli>(finished - sent).count();
    measurement.finished_s = std::chrono::duration<double>(finished - start).count();
}

json summarize(const std::vector<Measurement>& values, double elapsed) {
    std::vector<double> ttft, tpot, e2e, lag, itl;
    std::size_t success = 0, good = 0, tokens = 0;
    for (const auto& value : values) {
        lag.push_back(value.dispatch_lag_ms);
        if (!value.success) {
            continue;
        }
        ++success;
        good += static_cast<std::size_t>(value.within_slo);
        tokens += value.token_ids.size();
        ttft.push_back(value.ttft_ms);
        if (value.token_ids.size() > 1) {
            tpot.push_back(value.tpot_ms);
        }
        e2e.push_back(value.e2e_ms);
        for (std::size_t i = 1; i < value.token_times_ms.size(); ++i) {
            itl.push_back(value.token_times_ms[i] - value.token_times_ms[i - 1]);
        }
    }
    const auto percentiles = [](const auto& values) {
        if (values.empty()) {
            return json(nullptr);
        }
        return json{{"p50", percentile(values, 0.50)}, {"p95", percentile(values, 0.95)},
                    {"p99", percentile(values, 0.99)}};
    };
    return {{"requests", values.size()}, {"successful", success}, {"failed", values.size() - success},
            {"slo_compliant", good}, {"elapsed_s", elapsed}, {"successful_output_tokens", tokens},
            {"output_tokens_per_second", elapsed > 0 ? tokens / elapsed : 0},
            {"goodput_requests_per_second", elapsed > 0 ? good / elapsed : 0},
            {"ttft_ms", percentiles(ttft)}, {"mean_tpot_ms", percentiles(tpot)},
            {"inter_token_ms", percentiles(itl)}, {"e2e_ms", percentiles(e2e)},
            {"dispatch_lag_ms", percentiles(lag)}};
}

void make_trace(const Options& options, httplib::Client& client, const std::string& model) {
    const auto count = options.integer("--requests", 24, 1, 256);
    const auto rate = options.number("--rate", 4, 0, 10000);
    const auto seed = options.integer("--seed", 0, 0, 2147483647);
    const auto long_tokens = options.integer("--long-tokens", 128, 1, 4096);
    const auto short_tokens = options.integer("--short-tokens", 16, 1, 4096);
    const auto generated = options.integer("--max-tokens", 16, 1, 4096);
    const auto shared = options.has("--shared-prefix");
    auto file = output_file(options.get("--trace"));
    std::mt19937_64 random(static_cast<std::uint64_t>(seed));
    std::exponential_distribution<double> arrivals(rate > 0 ? static_cast<double>(rate) : 1.0);
    double arrival = 0;
    for (std::int64_t i = 0; i < count; ++i) {
        if (i > 0 && rate > 0) {
            arrival += arrivals(random);
        }
        const auto length = static_cast<std::size_t>(i % 4 == 0 ? long_tokens : short_tokens);
        const auto prefix = shared ? std::string("Shared memory and scheduling note. ") :
            "Request " + std::to_string(i) + ". Memory and scheduling note. ";
        const auto response = client.Post("/tokenize", json{{"text", prefix}}.dump(), "application/json");
        if (!response || response->status != 200) {
            throw std::runtime_error("trace tokenization failed");
        }
        const auto base = json::parse(response->body).at("tokens").get<std::vector<std::int32_t>>();
        if (base.empty()) {
            throw std::runtime_error("trace tokenizer returned an empty prompt");
        }
        std::vector<std::int32_t> prompt;
        while (prompt.size() < length) {
            prompt.insert(prompt.end(), base.begin(), base.end());
        }
        prompt.resize(length);
        file << json{{"request_id", "s" + std::to_string(seed) + "-r" + std::to_string(i)},
            {"arrival_s", arrival}, {"class_name", i % 4 == 0 ? "long" : "short"},
            {"ttft_slo_ms", options.integer("--ttft-slo-ms", 1500, 1, 300000)},
            {"tpot_slo_ms", options.integer("--tpot-slo-ms", 100, 1, 300000)},
            {"request", {{"model", model}, {"prompt", prompt}, {"max_tokens", generated},
                {"temperature", 0}, {"ignore_eos", true}, {"timeout_ms", 120000}}}}.dump() << '\n';
    }
    std::cout << "Trace: " << options.get("--trace") << " (" << count << " requests)\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options options(argc, argv, {"--port", "--trace", "--output", "--requests", "--rate", "--seed",
            "--long-tokens", "--short-tokens", "--max-tokens", "--ttft-slo-ms", "--tpot-slo-ms",
            "--run-id", "--trial", "--variant", "--trace-sha256", "--manifest-sha256",
            "--model-sha256", "--server-sha256", "--client-sha256", "--arrival-scale"},
            {"--help", "--make-trace", "--shared-prefix", "--no-warmup"});
        if (options.has("--help") || !options.has("--trace")) {
            std::cout << "llmserve-bench --trace WORKLOAD.jsonl --output REPORT.json [--port 8000] [--no-warmup]\n"
                         "               [--arrival-scale 1.0]\n"
                         "               [--run-id ID --trial N --variant NAME --trace-sha256 HASH\n"
                         "                --manifest-sha256 HASH --model-sha256 HASH --server-sha256 HASH --client-sha256 HASH]\n"
                         "llmserve-bench --make-trace --trace WORKLOAD.jsonl [--port 8000]\n"
                         "               [--requests 24] [--rate 4] [--seed 0] [--shared-prefix]\n"
                         "               [--long-tokens 128] [--short-tokens 16] [--max-tokens 16]\n"
                         "               [--ttft-slo-ms 1500] [--tpot-slo-ms 100]\n";
            return options.has("--help") ? 0 : 1;
        }
        const auto port = static_cast<int>(options.integer("--port", 8000, 1024, 65535));
        httplib::Client client("127.0.0.1", port);
        client.set_read_timeout(300);
        const auto model = get_json(client, "/metrics").at("model").get<std::string>();
        if (options.has("--make-trace")) {
            make_trace(options, client, model);
            return 0;
        }
        if (!options.has("--output")) {
            throw std::invalid_argument("--output is required for benchmark replay");
        }
        const auto run_id = options.get("--run-id");
        const auto variant = options.get("--variant");
        const auto trace_sha256 = options.get("--trace-sha256");
        const auto manifest_sha256 = options.get("--manifest-sha256");
        const auto model_sha256 = options.get("--model-sha256");
        const auto server_sha256 = options.get("--server-sha256");
        const auto client_sha256 = options.get("--client-sha256");
        const auto trial = options.integer("--trial", -1, -1, 1000000);
        const auto has_run_identity = !run_id.empty() || !variant.empty() || !trace_sha256.empty() ||
            !manifest_sha256.empty() || !model_sha256.empty() || !server_sha256.empty() ||
            !client_sha256.empty() || trial >= 0;
        if (has_run_identity && (run_id.empty() || variant.empty() || trace_sha256.empty() ||
            manifest_sha256.empty() || model_sha256.empty() || server_sha256.empty() ||
            client_sha256.empty() || trial < 0)) {
            throw std::invalid_argument("benchmark run identity fields must be supplied together");
        }
        if (has_run_identity) {
            for (const auto& hash : {trace_sha256, manifest_sha256, model_sha256, server_sha256, client_sha256}) {
                if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdef") != std::string::npos) {
                    throw std::invalid_argument("benchmark SHA-256 must use 64 lowercase hex digits");
                }
            }
        }
        std::ifstream input(options.get("--trace"), std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open workload trace");
        }
        const std::string raw_trace{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        std::istringstream trace_stream(raw_trace);
        std::vector<json> rows;
        const auto arrival_scale = options.number("--arrival-scale", 1, 0.000001, 10000);
        std::set<std::string> identifiers;
        std::string line;
        std::uint64_t digest = 14695981039346656037ull;
        for (const auto byte : raw_trace) {
            digest = (digest ^ static_cast<unsigned char>(byte)) * 1099511628211ull;
        }
        while (std::getline(trace_stream, line)) {
            auto row = json::parse(line);
            const auto arrival = row.at("arrival_s").get<double>();
            if (!std::isfinite(arrival) || arrival < 0 || arrival > 3600 ||
                (!rows.empty() && arrival < rows.back().at("arrival_s").get<double>()) ||
                !row.at("request_id").is_string() || !row.at("request").is_object() || rows.size() >= 256) {
                throw std::invalid_argument("invalid trace row or more than 256 requests");
            }
            if (row.at("request_id").get_ref<const std::string&>().empty() ||
                (row.contains("class_name") && !row["class_name"].is_string()) ||
                !identifiers.insert(row.at("request_id").get<std::string>()).second) {
                throw std::invalid_argument("invalid trace class or duplicate request ID");
            }
            for (const auto* key : {"ttft_slo_ms", "tpot_slo_ms"}) {
                if (row.contains(key) && (!row[key].is_number() ||
                    !std::isfinite(row[key].get<double>()) || row[key].get<double>() <= 0)) {
                    throw std::invalid_argument("trace SLO must be positive and finite");
                }
            }
            rows.push_back(std::move(row));
        }
        if (rows.empty()) {
            throw std::invalid_argument("empty workload trace");
        }
        for (auto& row : rows) {
            const auto scaled = row.at("arrival_s").get<double>() * arrival_scale;
            if (!std::isfinite(scaled) || scaled > 3600) {
                throw std::invalid_argument("scaled trace duration exceeds 3600 seconds");
            }
            row["arrival_s"] = scaled;
        }
        if (!options.has("--no-warmup")) {
            const auto warmup = client.Post("/v1/completions",
                json{{"prompt", "Hello"}, {"max_tokens", 8}, {"ignore_eos", true},
                     {"cache_namespace", "benchmark-warmup"}}.dump(), "application/json");
            if (!warmup || warmup->status != 200) {
                throw std::runtime_error("benchmark warmup failed");
            }
        }
        auto before = get_json(client, "/metrics");
        for (int i = 0; i < 100 && before.at("active_requests").get<std::size_t>() != 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            before = get_json(client, "/metrics");
        }
        if (before.at("outstanding_requests").get<std::size_t>() != 0 ||
            before.at("active_requests").get<std::size_t>() != 0) {
            throw std::runtime_error("benchmark requires an otherwise idle server");
        }
        const auto start = Clock::now() + std::chrono::seconds(1);
        const auto started_at_unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::vector<Measurement> results(rows.size());
        std::vector<std::jthread> workers;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            workers.emplace_back([&, i] { replay_one(rows[i], model, port, start, results[i]); });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        double elapsed = 0;
        json requests = json::array();
        std::map<std::string, std::vector<Measurement>> groups;
        for (const auto& result : results) {
            elapsed = std::max(elapsed, result.finished_s);
            groups[result.class_name].push_back(result);
            requests.push_back({{"id", result.id}, {"class_name", result.class_name},
                {"success", result.success}, {"within_slo", result.within_slo},
                {"error", result.error}, {"http_status", result.http_status},
                {"finish_reason", result.finish_reason}, {"sse_done_count", result.done_count},
                {"terminal_events", result.terminal_events},
                {"dispatch_lag_ms", result.dispatch_lag_ms}, {"ttft_ms", result.ttft_ms},
                {"mean_tpot_ms", result.token_ids.size() > 1 ? json(result.tpot_ms) : json(nullptr)},
                {"e2e_ms", result.e2e_ms}, {"finished_s", result.finished_s},
                {"token_times_ms", result.token_times_ms}, {"token_ids", result.token_ids},
                {"token_telemetry", result.token_telemetry},
                {"usage", result.usage}, {"server_timings", result.server_timings}});
        }
        json by_class;
        for (const auto& [name, values] : groups) {
            by_class[name] = summarize(values, elapsed);
        }
        auto after = get_json(client, "/metrics");
        for (int i = 0; i < 100 && (after.value("active_requests", 0) != 0 ||
             after.value("outstanding_requests", 0) != 0); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            after = get_json(client, "/metrics");
        }
        const auto summary = summarize(results, elapsed);
        json run_identity = nullptr;
        if (has_run_identity) {
            run_identity = {{"run_id", run_id}, {"trial", trial}, {"variant", variant},
                {"trace_sha256", trace_sha256}, {"manifest_sha256", manifest_sha256},
                {"model_sha256", model_sha256}, {"server_sha256", server_sha256},
                {"client_sha256", client_sha256}};
        }
        const json report{{"schema_version", 1}, {"benchmark", "llmserve-open-loop"}, {"clock", "steady_clock"},
            {"percentile_method", "linear"}, {"trace", options.get("--trace")},
            {"trace_fnv1a64", std::to_string(digest)}, {"warmup", !options.has("--no-warmup")},
            {"arrival_scale", arrival_scale},
            {"started_at_unix_ms", started_at_unix_ms},
            {"run_identity", run_identity},
            {"server_before", before}, {"server_after", after}, {"summary", summary},
            {"by_class", by_class}, {"requests", requests}};
        auto file = output_file(options.get("--output"));
        file << report.dump(2) << '\n';
        std::cout << summary.dump(2) << '\n';
        return summary.at("failed").get<std::size_t>() == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "llmserve-bench: " << error.what() << '\n';
        return 1;
    }
}
