#pragma once

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
             const std::function<void(std::size_t, std::size_t)>& work);

private:
    void consume();
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    std::atomic<std::size_t> next_{0};
    std::size_t count_ = 0;
    std::size_t grain_ = 1;
    std::size_t generation_ = 0;
    std::size_t remaining_ = 0;
    bool stop_ = false;
    std::function<void(std::size_t, std::size_t)> work_;
    std::exception_ptr error_;
};

} // namespace minillm
