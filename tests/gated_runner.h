#pragma once

#include "llmserve/model_runner.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>

namespace test {

class RunnerGate {
public:
    void wait_until_sampled() {
        std::unique_lock lock(mutex_);
        if (!cv_.wait_for(lock, std::chrono::seconds(30), [&] { return sampled_; })) {
            throw std::runtime_error("runner did not reach the first sample");
        }
    }

    void pause_after_sample() {
        std::unique_lock lock(mutex_);
        if (sampled_) {
            return;
        }
        sampled_ = true;
        cv_.notify_all();
        if (!cv_.wait_for(lock, std::chrono::seconds(30), [&] { return released_; })) {
            throw std::runtime_error("runner gate was not released");
        }
    }

    void release() {
        std::lock_guard lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool sampled_ = false;
    bool released_ = false;
};

class GatedRunner final : public llmserve::ModelRunner {
public:
    GatedRunner(std::unique_ptr<llmserve::ModelRunner> runner, std::shared_ptr<RunnerGate> gate)
        : runner_(std::move(runner)), gate_(std::move(gate)) {}

    const llmserve::ModelInfo& info() const noexcept override { return runner_->info(); }
    llmserve::BackendCapabilities capabilities() const noexcept override { return runner_->capabilities(); }
    std::vector<llmserve::Token> tokenize(std::string_view text) const override {
        return runner_->tokenize(text);
    }
    std::string token_piece(llmserve::Token token) const override { return runner_->token_piece(token); }
    bool is_eog(llmserve::Token token) const override { return runner_->is_eog(token); }
    std::vector<llmserve::Sample> execute(std::span<const llmserve::BatchToken> batch) override {
        auto samples = runner_->execute(batch);
        if (!samples.empty()) {
            // 首次 prefill 完成后暂停，允许下一轮所需的新请求确定地进入队列。
            gate_->pause_after_sample();
        }
        return samples;
    }
    std::vector<llmserve::Sample> execute_profiled(std::span<const llmserve::BatchToken> batch,
                                                 llmserve::RunnerTelemetry& profile) override {
        auto samples = runner_->execute_profiled(batch, profile);
        if (!samples.empty()) { gate_->pause_after_sample(); }
        return samples;
    }
    std::optional<llmserve::RunnerResources> resources() const noexcept override {
        return runner_->resources();
    }
    void copy_sequence(llmserve::SequenceId source, llmserve::SequenceId target,
                       std::size_t tokens) override {
        runner_->copy_sequence(source, target, tokens);
    }
    void clear_sequence(llmserve::SequenceId sequence) noexcept override {
        runner_->clear_sequence(sequence);
    }
    void synchronize() noexcept override { runner_->synchronize(); }

private:
    std::unique_ptr<llmserve::ModelRunner> runner_;
    std::shared_ptr<RunnerGate> gate_;
};

} // namespace test
