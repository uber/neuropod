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

#include "neuropod/multiprocess/control_messages.hh"
#include "neuropod/multiprocess/mq/ipc_message_queue.hh"

#include <mutex>

namespace neuropod
{

// Validates that state machine transitions are happening correctly
class TransitionVerifier
{
private:
    MessageType last_type_;
    bool        is_first_message_ = true;
    std::mutex  mutex_;

public:
    // Verifies that a state transition is allowed from the last state
    // to the current state
    void assert_transition_allowed(MessageType current_type);
};

class IPCControlChannel
{
private:
    using MessageQueue = IPCMessageQueue<MessageType>;

    // The name of the control queue;
    std::string control_queue_name_;

    // Control channels for communicating between the main process and worker process
    std::shared_ptr<MessageQueue> queue_;

    // Verifies that the state machine is operating as expected
    TransitionVerifier verifier_;

public:
    IPCControlChannel(const std::string &control_queue_name, ProcessType type);
    ~IPCControlChannel();

    // Utility to send a message with no content to a message queue
    // Note: this is threadsafe
    void send_message(MessageType type);

    // Utility to send a payload to a message queue
    // Note: this is threadsafe
    template <typename Payload>
    void send_message(MessageType payload_type, const Payload &payload)
    {
        verifier_.assert_transition_allowed(payload_type);
        queue_->send_message(payload_type, payload);
    }

    template <typename Payload>
    void send_message_move(MessageType payload_type, Payload payload)
    {
        verifier_.assert_transition_allowed(payload_type);
        queue_->send_message_move(payload_type, std::move(payload));
    }

    // Receive a message
    QueueMessage<MessageType> recv_message()
    {
        auto msg = queue_->recv_message();
        verifier_.assert_transition_allowed(msg.get_payload_type());

        return msg;
    }

    // Shutdown the control channel and cleanup the IPC queues
    void cleanup();
};

} // namespace neuropod
