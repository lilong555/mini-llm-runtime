#pragma once

#include "minillm/profile.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace minillm {

class ParallelExecutor {
public:
    explicit ParallelExecutor(std::size_t threads);
    ~ParallelExecutor();
    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;
    void run(std::size_t count, std::size_t grain,
             const std::function<void(std::size_t, std::size_t)>& work,
             ParallelProfile* profile = nullptr);

private:
    struct alignas(64) WorkerProfile {
        std::uint64_t start_ns = 0;
        std::uint64_t work_ns = 0;
        std::size_t chunks = 0;
    };
    void consume(std::size_t worker);
    std::vector<std::thread> threads_;
    std::vector<WorkerProfile> profiles_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    std::atomic<std::size_t> next_{0};
    std::size_t count_ = 0;
    std::size_t grain_ = 1;
    std::size_t generation_ = 0;
    std::size_t remaining_ = 0;
    bool stop_ = false;
    bool profiling_ = false;
    std::function<void(std::size_t, std::size_t)> work_;
    std::exception_ptr error_;
};

} // namespace minillm
