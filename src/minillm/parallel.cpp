#include "minillm/parallel.h"

#include <algorithm>
#include <stdexcept>

namespace minillm {

ParallelExecutor::ParallelExecutor(std::size_t threads) {
    if (threads == 0 || threads > 256) {
        throw std::invalid_argument("thread count must be in [1, 256]");
    }
    try {
        for (std::size_t i = 1; i < threads; ++i) {
            threads_.emplace_back([this] {
                std::size_t generation = 0;
                while (true) {
                    {
                        std::unique_lock lock(mutex_);
                        start_.wait(lock, [&] { return stop_ || generation_ != generation; });
                        if (stop_) {
                            return;
                        }
                        generation = generation_;
                    }
                    consume();
                    {
                        std::lock_guard lock(mutex_);
                        if (--remaining_ == 0) {
                            done_.notify_one();
                        }
                    }
                }
            });
        }
    } catch (...) {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        start_.notify_all();
        for (auto& thread : threads_) {
            thread.join();
        }
        throw;
    }
}

ParallelExecutor::~ParallelExecutor() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    start_.notify_all();
    for (auto& thread : threads_) {
        thread.join();
    }
}

void ParallelExecutor::consume() {
    try {
        while (true) {
            const auto begin = next_.fetch_add(grain_, std::memory_order_relaxed);
            if (begin >= count_) {
                return;
            }
            work_(begin, std::min(count_, begin + grain_));
        }
    } catch (...) {
        std::lock_guard lock(mutex_);
        if (!error_) {
            error_ = std::current_exception();
        }
    }
}

void ParallelExecutor::run(std::size_t count, std::size_t grain,
                            const std::function<void(std::size_t, std::size_t)>& work) {
    if (count == 0) {
        return;
    }
    if (grain == 0) {
        throw std::invalid_argument("parallel grain must be positive");
    }
    {
        std::lock_guard lock(mutex_);
        count_ = count;
        grain_ = grain;
        work_ = work;
        next_.store(0, std::memory_order_relaxed);
        error_ = nullptr;
        remaining_ = threads_.size();
        ++generation_;
    }
    start_.notify_all();
    consume();
    {
        std::unique_lock lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
        work_ = {};
        if (error_) {
            std::rethrow_exception(error_);
        }
    }
}

} // namespace minillm
