//
// Uber, Inc. (c) 2020
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>

namespace neuropod
{

// A threadsafe double buffered queue designed for one producer and one consumer
template <typename T>
class DoubleBufferedQueue
{
private:
    std::deque<T> front_;
    std::deque<T> back_;

    std::condition_variable cv_;
    std::mutex              front_mutex_;

    // If shutdown_ is set to true, the consumer will wake up and stop blocking
    // (if blocked)
    std::atomic_bool shutdown_{false};

public:
    DoubleBufferedQueue()  = default;
    ~DoubleBufferedQueue() = default;

    // Add an item to the queue
    // Can only be called from the producer thread
    void enqueue(T item)
    {
        std::unique_lock<std::mutex> lock(front_mutex_);

        bool front_was_empty = front_.size() == 0;
        front_.emplace_back(std::move(item));

        // A heuristic to see if we need to notify the worker thread
        // It is possible for us to notify when we don't need to, but
        // we will always notify if we went from "empty" to "not empty"
        if (front_was_empty)
        {
            // Notify consumer thread if one is waiting
            lock.unlock();
            cv_.notify_one();
        }
    }

    // Add all the items in `container` to the queue
    // Can only be called from the producer thread
    template <typename Container>
    void enqueue(Container container)
    {
        std::unique_lock<std::mutex> lock(front_mutex_);

        // A heuristic to see if we need to notify the worker thread
        // It is possible for us to notify when we don't need to, but
        // we will always notify if we went from "empty" to "not empty"
        bool front_was_empty = front_.size() == 0;
        for (auto &item : container)
        {
            front_.emplace_back(std::move(item));
        }

        if (front_was_empty)
        {
            // Notify consumer thread if one is waiting
            lock.unlock();
            cv_.notify_one();
        }
    }

    // Can only be called from the consumer thread
    // Returns `true` if an item was successfully returned
    bool dequeue(T &ret, bool do_wait = true)
    {
        // Return false if we shutdown this queue
        if (shutdown_)
        {
            return false;
        }

        // Check if we can get something from the back queue
        if (!back_.empty())
        {
            // Get the first item and return it
            ret = std::move(back_.front());
            back_.pop_front();

            return true;
        }

        // The back queue is empty
        // Check if the front queue has items in it
        std::unique_lock<std::mutex> front_lock(front_mutex_);
        if (!front_.empty())
        {
            // Swap front and back
            std::swap(front_, back_);

            // Unlock the front lock since we no longer need it
            front_lock.unlock();

            // Get the first item and return it
            ret = std::move(back_.front());
            back_.pop_front();

            return true;
        }

        if (do_wait)
        {
            // The front queue is empty
            // Wait until something gets added to it
            cv_.wait(front_lock, [&] { return shutdown_ || !front_.empty(); });

            if (shutdown_)
            {
                return false;
            }

            // Don't bother swapping the queues because there's likely only one item here
            // If there's more than one, we'll swap on the next call to `dequeue`

            // Get the first item from the front queue and return it
            ret = std::move(front_.front());
            front_.pop_front();

            return true;
        }

        return false;
    }

    // Can be called from any thread
    void shutdown()
    {
        shutdown_ = true;
        cv_.notify_one();
    }
};

} // namespace neuropod