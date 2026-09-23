#include "minillm/parallel.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace minillm
{
    namespace
    {
        std::uint64_t now_ns()
        {
            return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        }
    }

    ParallelExecutor::ParallelExecutor(std::size_t threads)
    {
        if (threads == 0 || threads > 256)
        {
            throw std::invalid_argument("thread count must be in [1, 256]");
        }
        profiles_.resize(threads);
        try
        {
            for (std::size_t i = 1; i < threads; ++i)
            {
                threads_.emplace_back([this, i]
                                      {
                std::size_t generation = 0;
                while (true) {
                    {
                        std::unique_lock lock(mutex_);
                        start_.wait(lock, [&] { return stop_ || generation_ != generation; });
                        if (stop_)
                            return;

                        generation = generation_;
                    }
                    consume(i);
                    {
                        std::lock_guard lock(mutex_);
                        if (--remaining_ == 0) {
                            done_.notify_one();
                        }
                    }
                } });
            }
        }
        catch (...)
        {
            {
                std::lock_guard lock(mutex_);
                stop_ = true;
            }
            start_.notify_all();
            for (auto &thread : threads_)
            {
                thread.join();
            }
            throw;
        }
    }

    ParallelExecutor::~ParallelExecutor()
    {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        start_.notify_all();
        for (auto &thread : threads_)
        {
            thread.join();
        }
    }

    void ParallelExecutor::consume(std::size_t worker)
    {
        const auto started = profiling_ ? now_ns() : 0;
        std::size_t chunks = 0;
        try
        {
            while (true)
            {
                const auto begin = next_.fetch_add(grain_, std::memory_order_relaxed);
                if (begin >= count_)
                {
                    break;
                }
                if (profiling_)
                    ++chunks;
                work_(begin, std::min(count_, begin + grain_));
            }
        }
        catch (...)
        {
            std::lock_guard lock(mutex_);
            if (!error_)
            {
                error_ = std::current_exception();
            }
        }
        if (profiling_)
        {
            profiles_[worker] = {started, now_ns() - started, chunks};
        }
    }

    void ParallelExecutor::run(std::size_t count, std::size_t grain,
                               const std::function<void(std::size_t, std::size_t)> &work,
                               ParallelProfile *profile)
    {
        const auto started = profile ? now_ns() : 0;
        if (profile)
        {
            *profile = {};
            profile->count = count;
            profile->grain = grain;
            profile->threads = profiles_.size();
        }
        if (count == 0)
        {
            if (profile)
            {
                profile->wall_ns = now_ns() - started;
                profile->completed = true;
            }
            return;
        }
        if (grain == 0)
        {
            throw std::invalid_argument("parallel grain must be positive");
        }
        {
            std::lock_guard lock(mutex_);
            count_ = count;
            grain_ = grain;
            work_ = work;
            profiling_ = profile != nullptr;
            next_.store(0, std::memory_order_relaxed);
            error_ = nullptr;
            remaining_ = threads_.size();
            ++generation_;
        }
        start_.notify_all();
        const auto dispatched = profile ? now_ns() : 0;
        consume(0);
        const auto wait_started = profile ? now_ns() : 0;
        {
            std::unique_lock lock(mutex_);
            done_.wait(lock, [&]
                       { return remaining_ == 0; });
            if (profile)
            {
                const auto finished = now_ns();
                profile->wall_ns = finished - started;
                profile->dispatch_ns = dispatched - started;
                profile->caller_work_ns = profiles_[0].work_ns;
                profile->caller_wait_ns = finished - wait_started;
                // 消费循环含任务分发和抢占，不等同于纯算术 CPU 时间。
                for (std::size_t i = 0; i < profiles_.size(); ++i)
                {
                    const auto &item = profiles_[i];
                    profile->chunks += item.chunks;
                    profile->participating_threads += item.chunks != 0 ? 1 : 0;
                    if (i != 0)
                    {
                        profile->worker_work_sum_ns += item.work_ns;
                        profile->worker_work_max_ns = std::max(profile->worker_work_max_ns, item.work_ns);
                        profile->worker_start_delay_max_ns = std::max(
                            profile->worker_start_delay_max_ns, item.start_ns - started);
                    }
                }
                profile->completed = !error_;
            }
            work_ = {};
            if (error_)
            {
                std::rethrow_exception(error_);
            }
        }
    }

} // namespace minillm
