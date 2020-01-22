//
// Uber, Inc. (c) 2020
//

#pragma once

#include "neuropod/internal/double_buffered_queue.hh"
#include "neuropod/internal/error_utils.hh"

#include <condition_variable>
#include <atomic>
#include <vector>

namespace neuropod
{

// A threadsafe double buffered priority queue designed for one producer and one consumer
template <typename T>
class DoubleBufferedPriorityQueue
{
private:
    size_t num_priorities_;

    // Items earlier in the vector have a higher priority
    std::vector<DoubleBufferedQueue<T>> queues;

    // A condition variable for notifying the consumer thread
    std::condition_variable cv_;
    std::mutex mutex_;

    size_t counter_ = 0;

    // If shutdown_ is set to true, the consumer will wake up and stop blocking
    // (if blocked)
    std::atomic_bool shutdown_{false};
public:
    DoubleBufferedPriorityQueue(size_t num_priorities)
        : num_priorities_(num_priorities), queues(num_priorities)
    {
    }

    ~DoubleBufferedPriorityQueue() = default;

    // Can only be called from the producer thread
    // Lower priorities are higher
    void enqueue(T item, size_t priority)
    {
        if (priority >= num_priorities_)
        {
            NEUROPOD_ERROR("priority must be < num_priorities provided in the constructor");
        }

        std::unique_lock<std::mutex> lk(mutex_);

        bool was_empty = counter_ == 0;
        counter_ += 1;

        // Add `item` to the appropriate queue
        queues.at(priority).enqueue(std::move(item));
        
        if (was_empty)
        {
            // Notify consumer thread if one is waiting 
            lk.unlock();
            cv_.notify_one();
        }
    }

    template <typename Container>
    void enqueue(Container &container, size_t priority)
    {
        if (priority >= num_priorities_)
        {
            NEUROPOD_ERROR("priority must be < num_priorities provided in the constructor");
        }

        std::unique_lock<std::mutex> lk(mutex_);
        bool was_empty = counter_ == 0;
        counter_ += container.size();

        // Add all the items to the appropriate queue
        queues.at(priority).enqueue(container);
        
        if (was_empty)
        {
            // Notify consumer thread if one is waiting 
            lk.unlock();
            cv_.notify_one();
        }
    }

    // Can only be called from the consumer thread
    // Returns `true` if an item was successfully returned
    // Will always return the highest priority item available
    bool dequeue(T &ret, size_t &priority, bool do_wait = true)
    {
        // Make sure we have at least one item before looping through queues
        size_t idx = 0;
        for (auto &queue : queues)
        {
            // Try to get something from the queue but don't wait
            if (queue.dequeue(ret, false))
            {
                // We successfully got an item
                std::unique_lock<std::mutex> lk(mutex_);
                --counter_;
                priority = idx;
                return true;
            }
            idx++;
        }

        if (do_wait)
        {
            {
                std::unique_lock<std::mutex> lk(mutex_);

                // Wait until something gets added to a queue
                cv_.wait(lk, [&]{
                    return shutdown_ || counter_ > 0;
                });

                if (shutdown_)
                {
                    return false;
                }
            }

            // Try again
            return dequeue(ret, priority, do_wait);
        }

        return false;
    }

    // Can only be called from the producer thread
    void shutdown()
    {
        shutdown_ = true;
        cv_.notify_one();
    }
};

} // namespace neuropod