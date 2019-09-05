//
// Uber, Inc. (c) 2019
//

#include "neuropods/multiprocess/message_utils.hh"

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/multiprocess/control_messages.hh"
#include "neuropods/multiprocess/shm_tensor.hh"

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
        const auto &block_id
            = std::dynamic_pointer_cast<NativeDataContainer<SHMBlockID>>(entry.second)->get_native_data();

        // Get the current index
        const auto current_index = msg.num_tensors;

        // Increment the number of tensors
        msg.num_tensors++;

        // Copy in the tensor name
        if (entry.first.length() >= 256)
        {
            NEUROPOD_ERROR("For the multiprocess backend, tensor names must have less than 256 characters. Tried using a tensor with name: " << entry.first);
        }

        strncpy(msg.tensor_name[current_index], entry.first.c_str(), 256);

        // Copy in the block ID
        static_assert(std::tuple_size<SHMBlockID>::value == sizeof(msg.tensor_id[0]), "The size of SHMBlockID should match the size of the IDs in control_message");
        memcpy(msg.tensor_id[current_index], block_id.data(), sizeof(msg.tensor_id[current_index]));

        // Send the message if needed
        if (msg.num_tensors == MAX_NUM_TENSORS_PER_MESSAGE)
        {
            queue.send(&msg, sizeof(control_message), 0);
            msg.num_tensors = 0;
        }
    }

    if (data.size() == 0 || msg.num_tensors != 0)
    {
        // Send the last message
        // (or the only message if we're attempting to send an empty map)
        queue.send(&msg, sizeof(control_message), 0);
    }
}

} // namespace neuropods
