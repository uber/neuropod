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

#include "neuropod/multiprocess/ipc_control_channel.hh"

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/internal/logging.hh"
#include "neuropod/multiprocess/control_messages.hh"
#include "neuropod/multiprocess/shm_tensor.hh"

namespace neuropod
{

void TransitionVerifier::assert_transition_allowed(MessageType current_type)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (current_type == SHUTDOWN || current_type == EXCEPTION)
    {
        // These messages are allowed at any time
        return;
    }

    // Special case for the first message
    if (is_first_message_ && current_type != LOAD_NEUROPOD)
    {
        NEUROPOD_ERROR("OPE: Invalid state transition. Expected LOAD_NEUROPOD as first state. Got {}", current_type);
    }

    // Using `set` instead of `unordered_set` because it doesn't require the type to be
    // hashable
    static const std::set<std::pair<MessageType, MessageType>> allowed_transitions = {
        std::make_pair(LOAD_NEUROPOD, LOAD_SUCCESS),
        std::make_pair(LOAD_SUCCESS, ADD_INPUT),
        std::make_pair(LOAD_SUCCESS, LOAD_NEUROPOD),
        std::make_pair(ADD_INPUT, INFER),
        std::make_pair(INFER, RETURN_OUTPUT),
        std::make_pair(RETURN_OUTPUT, ADD_INPUT),
        std::make_pair(RETURN_OUTPUT, LOAD_NEUROPOD),
    };

    if (!is_first_message_ &&
        allowed_transitions.find(std::make_pair(last_type_, current_type)) == allowed_transitions.end())
    {
        NEUROPOD_ERROR("OPE: Invalid state transition. Got transition from state {} to {}", last_type_, current_type);
    }

    last_type_        = current_type;
    is_first_message_ = false;
}

IPCControlChannel::IPCControlChannel(const std::string &control_queue_name, ProcessType type)
    : control_queue_name_(control_queue_name), queue_(std::make_shared<MessageQueue>(control_queue_name, type))
{
}

IPCControlChannel::~IPCControlChannel() = default;

void IPCControlChannel::send_message(MessageType type)
{
    verifier_.assert_transition_allowed(type);
    queue_->send_message(type);
}

void IPCControlChannel::cleanup()
{
    queue_.reset();
    cleanup_control_channels(control_queue_name_);
}

} // namespace neuropod
