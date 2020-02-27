//
// Uber, Inc. (c) 2020
//

#include "neuropod/multiprocess/ipc_message_queue.hh"

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
