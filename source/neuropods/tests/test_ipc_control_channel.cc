//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"
#include "neuropods/multiprocess/ipc_control_channel.hh"
#include "neuropods/multiprocess/shm_tensor.hh"
#include "timing_utils.hh"

TEST(test_ipc_control_channel, simple)
{
    // A tensor allocator that allocates tensors in shared memory
    std::unique_ptr<neuropods::NeuropodTensorAllocator> allocator =
        neuropods::stdx::make_unique<neuropods::DefaultTensorAllocator<neuropods::SHMNeuropodTensor>>();

    // Store tensors we allocate so they don't go out of scope
    neuropods::NeuropodValueMap sender_map;

    // Sample data
    constexpr size_t           num_items = 1024;
    const std::vector<int64_t> dims      = {2, 4, 8, 16};

    // TODO(vip): maybe dynamically generate a queue name?
    constexpr auto               queue_name = "neuropod_test_message_queue_simple";
    neuropods::IPCControlChannel main_control_channel(queue_name, neuropods::MAIN_PROCESS);
    neuropods::IPCControlChannel worker_control_channel(queue_name, neuropods::WORKER_PROCESS);

    // Allocate some tensors
    for (uint8_t i = 0; i < 16; i++)
    {
        const uint8_t some_data[num_items] = {i};

        // Allocate some memory and copy in data
        auto tensor = allocator->allocate_tensor<uint8_t>(dims);
        tensor->copy_from(some_data, num_items);

        sender_map[std::to_string(i)] = tensor;
    }

    // Send the tensors
    main_control_channel.send_message(neuropods::LOAD_NEUROPOD);
    main_control_channel.send_message(neuropods::ADD_INPUT, sender_map);
    main_control_channel.send_message(neuropods::INFER);

    // Receive the tensors
    neuropods::NeuropodValueMap recvd_map;
    for (int i = 0; i < 2; i++)
    {
        // Get a message
        neuropods::control_message received;
        worker_control_channel.recv_message(received);

        switch (i)
        {
        case 0:
            EXPECT_EQ(received.type, neuropods::LOAD_NEUROPOD);
            break;
        case 1:
            EXPECT_EQ(received.type, neuropods::ADD_INPUT);
            for (int i = 0; i < received.num_tensors; i++)
            {
                // Get the ID and create a tensor
                neuropods::SHMBlockID block_id;
                std::copy_n(received.tensor_id[i], block_id.size(), block_id.begin());
                auto shm_tensor = neuropods::tensor_from_id(block_id);

                recvd_map[received.tensor_name[i]] = shm_tensor;
            }
            break;
        default:
            EXPECT_EQ(received.type, neuropods::INFER);
            break;
        }
    }

    // Make sure the received tensors are what we expect
    EXPECT_EQ(recvd_map.size(), sender_map.size());
    for (const auto &item : recvd_map)
    {
        auto i      = static_cast<uint8_t>(std::stoi(item.first));
        auto tensor = item.second->as_typed_tensor<uint8_t>();

        // Make sure dims match
        auto actual_dims = tensor->get_dims();
        EXPECT_EQ(actual_dims, dims);

        // Make sure the data is what we expect
        const uint8_t expected_data[num_items] = {i};
        auto          actual_data              = tensor->get_raw_data_ptr();
        EXPECT_EQ(memcmp(actual_data, expected_data, num_items * sizeof(uint8_t)), 0);
    }

    // Cleanup
    main_control_channel.cleanup();
}

TEST(test_ipc_control_channel, no_tensors)
{
    // An empty map to send
    neuropods::NeuropodValueMap sender_map;

    // TODO(vip): maybe dynamically generate a queue name?
    constexpr auto               queue_name = "neuropod_test_message_queue_no_tensors";
    neuropods::IPCControlChannel main_control_channel(queue_name, neuropods::MAIN_PROCESS);
    neuropods::IPCControlChannel worker_control_channel(queue_name, neuropods::WORKER_PROCESS);

    // Send an empty map of tensors
    main_control_channel.send_message(neuropods::LOAD_NEUROPOD);
    main_control_channel.send_message(neuropods::ADD_INPUT, sender_map);
    main_control_channel.send_message(neuropods::INFER);

    // Receive the tensors
    for (int i = 0; i < 2; i++)
    {
        // Get a message
        neuropods::control_message received;
        worker_control_channel.recv_message(received);

        switch (i)
        {
        case 0:
            EXPECT_EQ(received.type, neuropods::LOAD_NEUROPOD);
            break;
        case 1:
            EXPECT_EQ(received.type, neuropods::ADD_INPUT);
            EXPECT_EQ(received.num_tensors, 0);
            break;
        default:
            EXPECT_EQ(received.type, neuropods::INFER);
            break;
        }
    }

    // Cleanup
    main_control_channel.cleanup();
}

TEST(test_ipc_control_channel, invalid_transition)
{
    // TODO(vip): maybe dynamically generate a queue name?
    // Open the control channels
    const std::string            control_queue_name = "test_multiprocess_worker_invalid_transition";
    neuropods::IPCControlChannel control_channel(control_queue_name, neuropods::MAIN_PROCESS);

    // Send a message (that is invalid as the first message)
    // We should get a failure here
    EXPECT_ANY_THROW(control_channel.send_message(neuropods::INFER));

    // Cleanup
    control_channel.cleanup();
}
