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

#include "neuropod/multiprocess/mq/wire_format.hh"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace neuropod
{

namespace detail
{

// The interval at which each process should send a heartbeat
constexpr int HEARTBEAT_INTERVAL_MS = 2000;

// The timeout for a process to receive a message on a queue
// This is generous since heartbeats are sent every 2 seconds (defined above)
constexpr int MESSAGE_TIMEOUT_MS = 5000;

// Ensure the timeout is larger than the heartbeat interval
static_assert(MESSAGE_TIMEOUT_MS > HEARTBEAT_INTERVAL_MS, "Message timeout must be larger than the heartbeat interval");

// Starts a new thread to send a heartbeat
class HeartbeatController
{
private:
    // State needed for the heartbeat thread
    std::atomic_bool        send_heartbeat_{true};
    std::condition_variable cv_;
    std::mutex              mutex_;
    std::thread             heartbeat_thread_;

public:
    template <typename MessageQueue>
    HeartbeatController(MessageQueue &control_channel)
        : send_heartbeat_(true), heartbeat_thread_([this, &control_channel]() {
              // Send a heartbeat periodically
              while (send_heartbeat_)
              {
                  // Send a heartbeat message
                  typename MessageQueue::WireFormat msg;
                  msg.type = detail::HEARTBEAT;
                  control_channel.send_message(msg);

                  // Using a condition variable lets us wake up while we're waiting
                  std::unique_lock<std::mutex> lk(mutex_);
                  cv_.wait_for(
                      lk, std::chrono::milliseconds(HEARTBEAT_INTERVAL_MS), [&] { return send_heartbeat_ != true; });
              }
          })
    {
    }

    ~HeartbeatController()
    {
        // Join the heartbeat thread
        {
            std::lock_guard<std::mutex> lk(mutex_);
            send_heartbeat_ = false;
        }

        cv_.notify_all();
        heartbeat_thread_.join();
    }
};

} // namespace detail

} // namespace neuropod
