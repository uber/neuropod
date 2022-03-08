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

#include "neuropod/multiprocess/mq/ipc_message_queue.hh"

namespace neuropod
{

namespace detail
{

// Used to generate IDs for messages
std::atomic_uint64_t msg_counter;

} // namespace detail

void cleanup_control_channels(const std::string &control_queue_name)
{
    // Delete the control channels
    ipc::message_queue::remove(("neuropod_" + control_queue_name + "_tw").c_str());
    ipc::message_queue::remove(("neuropod_" + control_queue_name + "_fw").c_str());
}

} // namespace neuropod
