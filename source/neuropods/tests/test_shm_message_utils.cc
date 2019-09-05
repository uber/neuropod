//
// Uber, Inc. (c) 2019
//

#include "timing_utils.hh"
#include "neuropods/multiprocess/message_utils.hh"
#include "neuropods/multiprocess/shm_tensor.hh"

#include "gtest/gtest.h"

TEST(test_shm_message_utils, simple)
{
    // A tensor allocator that allocates tensors in shared memory
    std::unique_ptr<neuropods::NeuropodTensorAllocator> allocator = neuropods::stdx::make_unique<neuropods::DefaultTensorAllocator<neuropods::SHMNeuropodTensor>>();

    // Store tensors we allocate so they don't go out of scope
    neuropods::NeuropodValueMap sender_map;

    // Sample data
    constexpr size_t num_items = 1024;
    const std::vector<int64_t> dims = {2, 4, 8, 16};

    // TODO(vip): maybe dynamically generate a queue name?
    constexpr auto queue_name = "neuropod_test_message_queue_simple";
    ipc::message_queue::remove(queue_name);
    ipc::message_queue send_queue(ipc::create_only, queue_name, 100, sizeof(neuropods::control_message));
    ipc::message_queue recv_queue(ipc::open_only, queue_name);

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
    neuropods::send_message(send_queue, neuropods::ADD_INPUT, sender_map);
    neuropods::send_message(send_queue, neuropods::INFER);

    // Receive the tensors
    neuropods::NeuropodValueMap recvd_map;
    while (true)
    {
        // Get a message
        neuropods::control_message received;
        size_t received_size;
        unsigned int priority;
        recv_queue.receive(&received, sizeof(neuropods::control_message), received_size, priority);

        if (received.type == neuropods::INFER)
        {
            break;
        }

        EXPECT_EQ(received.type, neuropods::ADD_INPUT);

        for (int i = 0; i < received.num_tensors; i++)
        {
            // Get the uuid and create a tensor
            neuropods::SHMBlockID block_id;
            std::copy_n(received.tensor_id[i], block_id.size(), block_id.begin());
            auto shm_tensor = neuropods::tensor_from_id(block_id);

            recvd_map[received.tensor_name[i]] = shm_tensor;
        }
    }

    // Make sure the received tensors are what we expect
    EXPECT_EQ(recvd_map.size(), sender_map.size());
    for (const auto &item : recvd_map)
    {
        auto i = static_cast<uint8_t>(std::stoi(item.first));
        auto tensor = item.second->as_typed_tensor<uint8_t>();

        // Make sure dims match
        auto actual_dims = tensor->get_dims();
        EXPECT_EQ(actual_dims, dims);

        // Make sure the data is what we expect
        const uint8_t expected_data[num_items] = {i};
        auto actual_data = tensor->get_raw_data_ptr();
        EXPECT_EQ(memcmp(actual_data, expected_data, num_items * sizeof(uint8_t)), 0);
    }

    // Cleanup
    ipc::message_queue::remove(queue_name);
}

TEST(test_shm_message_utils, no_tensors)
{
    // An empty map to send
    neuropods::NeuropodValueMap sender_map;

    // TODO(vip): maybe dynamically generate a queue name?
    constexpr auto queue_name = "neuropod_test_message_queue_no_tensors";
    ipc::message_queue::remove(queue_name);
    ipc::message_queue send_queue(ipc::create_only, queue_name, 100, sizeof(neuropods::control_message));
    ipc::message_queue recv_queue(ipc::open_only, queue_name);

    // Send an empty map of tensors
    neuropods::send_message(send_queue, neuropods::ADD_INPUT, sender_map);
    neuropods::send_message(send_queue, neuropods::INFER);

    // Receive the tensors
    for (int i = 0; i < 2; i++)
    {
        // Get a message
        neuropods::control_message received;
        size_t received_size;
        unsigned int priority;
        recv_queue.receive(&received, sizeof(neuropods::control_message), received_size, priority);

        if (i == 0)
        {
            EXPECT_EQ(received.type, neuropods::ADD_INPUT);
            EXPECT_EQ(received.num_tensors, 0);
        }
        else
        {
            EXPECT_EQ(received.type, neuropods::INFER);
        }
    }

    // Cleanup
    ipc::message_queue::remove(queue_name);
}
