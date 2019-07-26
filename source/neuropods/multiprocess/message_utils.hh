//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/multiprocess/multiprocess.hh"

#include "neuropods/multiprocess/control_messages.hh"
#include "neuropods/multiprocess/shm_tensor.hh"

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/uuid/uuid.hpp>

namespace ipc = boost::interprocess;

namespace neuropods
{

// Utility to send a message with no content to a message queue
void send_message(ipc::message_queue &queue, MessageType type)
{
    control_message msg;
    msg.type = type;
    queue.send(&msg, sizeof(control_message), 0);
}

// Utility to send a NeuropodValueMap to a message queue
void send_message(ipc::message_queue &queue, MessageType type, const NeuropodValueMap &data)
{
    control_message msg;
    msg.type = type;
    msg.num_tensors = 0;

    for (const auto &entry : data)
    {
        const auto &uuid
            = std::dynamic_pointer_cast<NativeDataContainer<boost::uuids::uuid&>>(entry.second)->get_native_data();

        // Get the current index
        auto current_index = msg.num_tensors;

        // Increment the number of tensors
        msg.num_tensors++;

        // Copy in the tensor name
        // TODO(vip): Max len check
        strncpy(msg.tensor_name[current_index], entry.first.c_str(), 256);

        // Copy in the UUID
        std::copy(uuid.begin(), uuid.end(), msg.tensor_uuid[current_index]);

        // Send the message if needed
        if (msg.num_tensors == MAX_NUM_TENSORS_PER_MESSAGE)
        {
            queue.send(&msg, sizeof(control_message), 0);
            msg.num_tensors = 0;
        }
    }

    if (msg.num_tensors != 0)
    {
        // Send the last message
        queue.send(&msg, sizeof(control_message), 0);
    }
}

} // namespace neuropods
