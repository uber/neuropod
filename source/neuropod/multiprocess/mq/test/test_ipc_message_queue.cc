/* Copyright (c) 2020 UATC, LLC

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

#include "gtest/gtest.h"
#include "neuropod/multiprocess/control_messages.hh"
#include "neuropod/multiprocess/mq/ipc_message_queue.hh"
#include "neuropod/multiprocess/shm_tensor.hh"

// TODO(vip): These tests are basically identical to the ones in `test_ipc_control_channel`
// Delete these tests once ipc_control_channel is using ipc_message_queue
TEST(test_ipc_message_queue, simple)
{
    // TODO(vip): maybe dynamically generate a queue name?
    constexpr auto queue_name = "neuropod_test_message_queue_simple";
    {
        // A tensor allocator that allocates tensors in shared memory
        std::unique_ptr<neuropod::NeuropodTensorAllocator> allocator =
            neuropod::stdx::make_unique<neuropod::DefaultTensorAllocator<neuropod::SHMNeuropodTensor>>();

        // Store tensors we allocate so they don't go out of scope
        neuropod::NeuropodValueMap sender_map;

        // Sample data
        constexpr size_t           num_items = 1024;
        const std::vector<int64_t> dims      = {2, 4, 8, 16};

        auto main_control_channel =
            std::make_shared<neuropod::IPCMessageQueue<neuropod::MessageType>>(queue_name, neuropod::MAIN_PROCESS);
        auto worker_control_channel =
            std::make_shared<neuropod::IPCMessageQueue<neuropod::MessageType>>(queue_name, neuropod::WORKER_PROCESS);

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
        main_control_channel->send_message(neuropod::LOAD_NEUROPOD);
        main_control_channel->send_message(neuropod::LOAD_SUCCESS);
        main_control_channel->send_message_move(neuropod::ADD_INPUT, sender_map);
        main_control_channel->send_message(neuropod::INFER);

        // Receive the tensors
        neuropod::NeuropodValueMap recvd_map;
        for (int i = 0; i < 4; i++)
        {
            // Get a message
            auto received = worker_control_channel->recv_message();
            auto msg_type = received.get_payload_type();

            switch (i)
            {
            case 0:
                EXPECT_EQ(msg_type, neuropod::LOAD_NEUROPOD);
                break;
            case 1:
                EXPECT_EQ(msg_type, neuropod::LOAD_SUCCESS);
                break;
            case 2:
                EXPECT_EQ(msg_type, neuropod::ADD_INPUT);
                received.get(recvd_map);
                break;
            default:
                EXPECT_EQ(msg_type, neuropod::INFER);
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
    }

    // Cleanup
    neuropod::cleanup_control_channels(queue_name);
}

TEST(test_ipc_message_queue, no_tensors)
{
    // TODO(vip): maybe dynamically generate a queue name?
    constexpr auto queue_name = "neuropod_test_message_queue_no_tensors";
    {
        // An empty map to send
        neuropod::NeuropodValueMap sender_map;

        auto main_control_channel =
            std::make_shared<neuropod::IPCMessageQueue<neuropod::MessageType>>(queue_name, neuropod::MAIN_PROCESS);
        auto worker_control_channel =
            std::make_shared<neuropod::IPCMessageQueue<neuropod::MessageType>>(queue_name, neuropod::WORKER_PROCESS);

        // Send an empty map of tensors
        main_control_channel->send_message(neuropod::LOAD_NEUROPOD);
        main_control_channel->send_message(neuropod::LOAD_SUCCESS);
        main_control_channel->send_message_move(neuropod::ADD_INPUT, sender_map);
        main_control_channel->send_message(neuropod::INFER);

        // Receive the tensors
        for (int i = 0; i < 4; i++)
        {
            // Get a message
            auto received = worker_control_channel->recv_message();
            auto msg_type = received.get_payload_type();

            switch (i)
            {
            case 0:
                EXPECT_EQ(msg_type, neuropod::LOAD_NEUROPOD);
                break;
            case 1:
                EXPECT_EQ(msg_type, neuropod::LOAD_SUCCESS);
                break;
            case 2: {
                // We need a new scope here because of the `tmp` variable declaration
                EXPECT_EQ(msg_type, neuropod::ADD_INPUT);
                neuropod::NeuropodValueMap tmp;
                received.get(tmp);
                EXPECT_EQ(tmp.size(), 0);
                break;
            }
            default:
                EXPECT_EQ(msg_type, neuropod::INFER);
                break;
            }
        }
    }

    // Cleanup
    neuropod::cleanup_control_channels(queue_name);
}
