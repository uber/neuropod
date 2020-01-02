//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"
#include "neuropod/multiprocess/ipc_control_channel.hh"
#include "neuropod/multiprocess/shm_tensor.hh"

TEST(test_ipc_control_channel, simple)
{
    // A tensor allocator that allocates tensors in shared memory
    std::unique_ptr<neuropod::NeuropodTensorAllocator> allocator =
        neuropod::stdx::make_unique<neuropod::DefaultTensorAllocator<neuropod::SHMNeuropodTensor>>();

    // Store tensors we allocate so they don't go out of scope
    neuropod::NeuropodValueMap sender_map;

    // Sample data
    constexpr size_t           num_items = 1024;
    const std::vector<int64_t> dims      = {2, 4, 8, 16};

    // TODO(vip): maybe dynamically generate a queue name?
    constexpr auto              queue_name = "neuropod_test_message_queue_simple";
    neuropod::IPCControlChannel main_control_channel(queue_name, neuropod::MAIN_PROCESS);
    neuropod::IPCControlChannel worker_control_channel(queue_name, neuropod::WORKER_PROCESS);

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
    main_control_channel.send_message(neuropod::LOAD_NEUROPOD);
    main_control_channel.send_message(neuropod::ADD_INPUT, sender_map);
    main_control_channel.send_message(neuropod::INFER);

    // Receive the tensors
    neuropod::NeuropodValueMap recvd_map;
    for (int i = 0; i < 2; i++)
    {
        // Get a message
        neuropod::control_message received;
        worker_control_channel.recv_message(received);

        switch (i)
        {
        case 0:
            EXPECT_EQ(received.type, neuropod::LOAD_NEUROPOD);
            break;
        case 1:
            EXPECT_EQ(received.type, neuropod::ADD_INPUT);
            for (int i = 0; i < received.num_tensors; i++)
            {
                // Get the ID and create a tensor
                neuropod::SHMBlockID block_id;
                std::copy_n(received.tensor_id[i], block_id.size(), block_id.begin());
                auto shm_tensor = neuropod::tensor_from_id(block_id);

                recvd_map[received.tensor_name[i]] = shm_tensor;
            }
            break;
        default:
            EXPECT_EQ(received.type, neuropod::INFER);
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
    neuropod::NeuropodValueMap sender_map;

    // TODO(vip): maybe dynamically generate a queue name?
    constexpr auto              queue_name = "neuropod_test_message_queue_no_tensors";
    neuropod::IPCControlChannel main_control_channel(queue_name, neuropod::MAIN_PROCESS);
    neuropod::IPCControlChannel worker_control_channel(queue_name, neuropod::WORKER_PROCESS);

    // Send an empty map of tensors
    main_control_channel.send_message(neuropod::LOAD_NEUROPOD);
    main_control_channel.send_message(neuropod::ADD_INPUT, sender_map);
    main_control_channel.send_message(neuropod::INFER);

    // Receive the tensors
    for (int i = 0; i < 2; i++)
    {
        // Get a message
        neuropod::control_message received;
        worker_control_channel.recv_message(received);

        switch (i)
        {
        case 0:
            EXPECT_EQ(received.type, neuropod::LOAD_NEUROPOD);
            break;
        case 1:
            EXPECT_EQ(received.type, neuropod::ADD_INPUT);
            EXPECT_EQ(received.num_tensors, 0);
            break;
        default:
            EXPECT_EQ(received.type, neuropod::INFER);
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
    const std::string           control_queue_name = "test_multiprocess_worker_invalid_transition";
    neuropod::IPCControlChannel control_channel(control_queue_name, neuropod::MAIN_PROCESS);

    // Send a message (that is invalid as the first message)
    // We should get a failure here
    EXPECT_ANY_THROW(control_channel.send_message(neuropod::INFER));

    // Cleanup
    control_channel.cleanup();
}
