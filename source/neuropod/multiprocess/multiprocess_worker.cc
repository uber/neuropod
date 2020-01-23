//
// Uber, Inc. (c) 2019
//

#include "neuropod/internal/worker_thread.hh"
#include "neuropod/multiprocess/control_messages.hh"
#include "neuropod/multiprocess/ipc_control_channel.hh"
#include "neuropod/multiprocess/shm_tensor.hh"
#include "neuropod/multiprocess/tensor_utils.hh"
#include "neuropod/neuropod.hh"

#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace neuropod
{

// The main loop for a worker that runs a neuropod
void multiprocess_worker_loop(const std::string &control_queue_name)
{
    // Open the control channels
    IPCControlChannel control_channel(control_queue_name, WORKER_PROCESS);

    // A pointer to a neuropod (that will be loaded)
    std::unique_ptr<Neuropod>                neuropod;
    std::shared_ptr<NeuropodTensorAllocator> allocator;

    // A map to store the inputs
    NeuropodValueMap inputs;

    // The last outputs
    // We need to keep these around so there isn't a race condition when returning
    // data back to the main process
    NeuropodValueMap last_outputs;

    // Send a heartbeat periodically
    WorkerThreadWithLoop heartbeat_controller([&control_channel] {
        // send_message is threadsafe so we don't need synchronization here
        control_channel.send_message(HEARTBEAT);
        std::this_thread::sleep_for(std::chrono::milliseconds(HEARTBEAT_INTERVAL_MS));
    });

    // A stuct to wrap an id and tensor
    struct TensorWrapper
    {
        SHMBlockID                     id;
        std::shared_ptr<NeuropodValue> tensor;
    };

    // A map from an ID to a tensor name
    std::unordered_map<SHMBlockID, std::string> id_to_name_map;

    WorkerThreadWithInputAndOutputQueue<SHMBlockID, TensorWrapper> tensor_loader(
        [&allocator](SHMBlockID &block_id, TensorWrapper &output) {
            output.id = block_id;

            // Load the tensor
            auto shm_tensor = tensor_from_id(block_id);

            // Wrap in a tensor type that this neuropod expects
            output.tensor = wrap_existing_tensor(*allocator, shm_tensor);

            // We returned an item so return true
            return true;
        });

    while (true)
    {
        // Get a message
        control_message received;
        control_channel.recv_message(received);

        if (received.type == LOAD_NEUROPOD)
        {
            // Load a neuropod
            neuropod  = stdx::make_unique<Neuropod>(received.neuropod_path);
            allocator = neuropod->get_tensor_allocator();
            inputs.clear();
            last_outputs.clear();
        }
        else if (received.type == ADD_INPUT)
        {
            std::vector<SHMBlockID> block_ids(received.num_tensors);
            for (int i = 0; i < received.num_tensors; i++)
            {
                // Get the ID and create a tensor
                SHMBlockID &block_id = block_ids[i];
                std::copy_n(received.tensor_id[i], block_id.size(), block_id.begin());

                // Store the tensor name
                std::string tensor_name  = received.tensor_name[i];
                id_to_name_map[block_id] = tensor_name;
            }

            // Add them to the queue for async loading
            tensor_loader.enqueue(block_ids);
        }
        else if (received.type == INFER)
        {
            // Get all the inputs
            while (!id_to_name_map.empty())
            {
                TensorWrapper item;
                bool          success = tensor_loader.dequeue(item);
                if (!success)
                {
                    break;
                }

                inputs[id_to_name_map.at(item.id)] = std::move(item.tensor);
                id_to_name_map.erase(item.id);
            }

            // Run inference
            auto outputs = neuropod->infer(inputs);

            // Turn these "native" tensors into shm tensors
            for (const auto &entry : *outputs)
            {
                // Unfortunately, this requires a copy (done within SHMNeuropodTensor)
                auto shm_tensor =
                    wrap_existing_tensor<SHMNeuropodTensor>(std::dynamic_pointer_cast<NeuropodTensor>(entry.second));

                // This ensures that the tensor stays around long enough for the other process to load it
                last_outputs[entry.first] = shm_tensor;
            }

            control_channel.send_message(RETURN_OUTPUT, last_outputs);

            // Let the main process know that we're done
            control_channel.send_message(END_OUTPUT);

            // Clean up any unused shm tensors that haven't been reused
            shm_allocator.free_unused_shm_blocks();

            // Empty the inputs set. This is done after sending outputs back to the main process
            // because this takes a nontrivial amount of time
            inputs.clear();
        }
        else if (received.type == INFER_COMPLETE)
        {
            // The main process loaded our output tensors
            // We no longer need to maintain references to these tensors
            last_outputs.clear();
        }
        else if (received.type == SHUTDOWN)
        {
            break;
        }
    }
}

} // namespace neuropod
