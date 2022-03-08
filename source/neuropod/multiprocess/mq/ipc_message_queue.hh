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

#include "neuropod/internal/blocking_spsc_queue.hh"
#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/memory_utils.hh"
#include "neuropod/multiprocess/mq/heartbeat.hh"
#include "neuropod/multiprocess/mq/wire_format.hh"

#include <boost/interprocess/ipc/message_queue.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <unordered_map>

namespace ipc = boost::interprocess;

namespace neuropod
{

// Forward declare IPCMessageQueue
template <typename>
class IPCMessageQueue;

// This is the user-facing queue message type
// UserPayloadType should be an enum that specifies types of payloads
template <typename UserPayloadType>
class QueueMessage
{
private:
    // A pointer to the underlying data
    std::shared_ptr<detail::WireFormat<UserPayloadType>> data_;

    // Note: The constructor is private and only used in `IPCMessageQueue`
    template <typename>
    friend class IPCMessageQueue;

    // Constructor used when receiving messages
    QueueMessage(std::shared_ptr<detail::WireFormat<UserPayloadType>> data) : data_(std::move(data)) {}

public:
    ~QueueMessage() = default;

    // Get a payload of type `Payload` from this message
    template <typename Payload>
    void get(Payload &out)
    {
        detail::deserialize_payload(*data_, out);
    }

    // Get the type of the user-defined payload included in this message
    // This should only be used when it is known that this message contains
    // a user-defined payload
    UserPayloadType get_payload_type() { return data_->payload_type; }
};

// The type of the process creating the message queue
enum ProcessType
{
    WORKER_PROCESS,
    MAIN_PROCESS,
};

// A bidirectional IPC message queue that supports cross-process moves or copies of payloads.
// Includes an implementation of heartbeats and message acknowledgement (in the form of DONE messages)
// This class starts a thread for reading from the underlying `recv_queue` and uses a `HeartbeatController`
// to start a thread for sending heartbeats.
template <typename UserPayloadType>
class IPCMessageQueue : public std::enable_shared_from_this<IPCMessageQueue<UserPayloadType>>
{
private:
    using WireFormat = detail::WireFormat<UserPayloadType>;

    // A queue to store output messages received by the read thread
    BlockingSPSCQueue<std::unique_ptr<WireFormat>> out_queue_;

    // Internal IPC queues to communicate with the other process
    std::string                         control_queue_name_;
    std::unique_ptr<ipc::message_queue> send_queue_;
    std::unique_ptr<ipc::message_queue> recv_queue_;

    // Responsible for periodically sending a heartbeat
    friend class detail::HeartbeatController;
    std::unique_ptr<detail::HeartbeatController> heartbeat_controller_;

    // Responsible for keeping things in scope during cross-process moves
    std::unique_ptr<detail::TransferrableController> transferrable_controller_;

    // Whether or not a shutdown is in progress
    bool shutdown_started_ = false;

    // If we lost the heartbeat from the other process
    std::atomic_bool lost_heartbeat_;

    // A thread that handles incoming messages
    std::thread read_worker_;

    // The worker loop for the message reading thread
    void read_worker_loop();

    // Send a message to the other process
    void send_message(const WireFormat &msg);

    // Throw an error if we lost communication with the other process
    void throw_if_lost_heartbeat();

public:
    IPCMessageQueue(const std::string &control_queue_name, ProcessType type);

    ~IPCMessageQueue();

    // Send a message with a payload
    // Note: this is threadsafe
    template <typename Payload>
    void send_message(UserPayloadType payload_type, const Payload &payload);

    // Send a message with a payload and ensure `payload` stays in
    // scope until the other process is done using the message.
    // Note: this is threadsafe
    template <typename Payload>
    void send_message_move(UserPayloadType payload_type, Payload payload);

    // Send a message with just a payload_type
    // Note: this is threadsafe
    void send_message(UserPayloadType payload_type);

    // Get a message. Blocks if the queue is empty.
    // Note: this is _NOT_ threadsafe. There should only be one thread calling `recv_message`
    // at a time.
    QueueMessage<UserPayloadType> recv_message();
};

// Cleanup control channels for the queue with name `control_queue_name`
void cleanup_control_channels(const std::string &control_queue_name);

} // namespace neuropod

#include "neuropod/multiprocess/mq/ipc_message_queue_impl.hh"
