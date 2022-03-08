/* Copyright (c) 2020 The Neuropod Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace neuropod
{

// A bounded blocking single producer single consumer (SPSC) queue
template <typename T>
class BlockingSPSCQueue
{
private:
    std::queue<T>           queue_;
    std::condition_variable full_cv_;
    std::condition_variable empty_cv_;
    std::mutex              mutex_;

    size_t capacity_;

public:
    BlockingSPSCQueue(size_t capacity) : capacity_(capacity) {}
    ~BlockingSPSCQueue() = default;

    bool try_emplace(T &&item)
    {
        bool success = false;

        // Lock the mutex and add to the queue if we have capacity
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (queue_.size() < capacity_)
            {
                queue_.emplace(std::forward<T>(item));
                success = true;
            }
        }

        // Notify any waiting read threads if we need to
        if (success)
        {
            empty_cv_.notify_all();
        }

        return success;
    }

    void emplace(T &&item)
    {
        // Lock the mutex and add to the queue (or wait until we have capacity)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (queue_.size() >= capacity_)
            {
                full_cv_.wait(lock, [&] { return queue_.size() < capacity_; });
            }

            queue_.emplace(std::forward<T>(item));
        }

        // Notify any waiting read threads
        empty_cv_.notify_all();
    }

    void pop(T &item)
    {
        // Lock the mutex and get an item from the queue (or wait until we have an item)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (queue_.empty())
            {
                empty_cv_.wait(lock, [&] { return !queue_.empty(); });
            }

            item = std::move(queue_.front());
            queue_.pop();
        }

        // Notify any waiting write threads
        full_cv_.notify_all();
    }
};

} // namespace neuropod
