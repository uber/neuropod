//
// Uber, Inc. (c) 2019
//

#include "neuropod/internal/logging.hh"
#include "neuropod/multiprocess/control_messages.hh"
#include "neuropod/multiprocess/heartbeat_controller.hh"
#include "neuropod/multiprocess/ipc_control_channel.hh"
#include "neuropod/multiprocess/shm_tensor.hh"
#include "neuropod/multiprocess/tensor_utils.hh"
#include "neuropod/neuropod.hh"

#include <atomic>
#include <condition_variable>
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

    // A vector of requested outputs
    std::vector<std::string> requested_outputs;

    // The last outputs
    // We need to keep these around so there isn't a race condition when returning
    // data back to the main process
    NeuropodValueMap last_outputs;

    // Send a heartbeat periodically
    HeartbeatController heartbeat_controller(control_channel, HEARTBEAT_INTERVAL_MS);

    while (true)
    {
        // Get a message with a timeout
        control_message received;
        bool            successful_read = control_channel.recv_message(received, MESSAGE_TIMEOUT_MS);
        if (!successful_read)
        {
            // We timed out. This usually means the main process died.
            NEUROPOD_ERROR("Timed out waiting for a heartbeat from the main process. "
                           "Didn't receive a message in {}ms, but expected a heartbeat every {}ms.",
                           MESSAGE_TIMEOUT_MS,
                           HEARTBEAT_INTERVAL_MS);
        }

        if (received.type == LOAD_NEUROPOD)
        {
            // Load a neuropod
            neuropod  = stdx::make_unique<Neuropod>(received.neuropod_path);
            allocator = neuropod->get_tensor_allocator();
            inputs.clear();
            last_outputs.clear();
            control_channel.send_message(LOAD_SUCCESS);
        }
        else if (received.type == ADD_INPUT)
        {
            for (int i = 0; i < received.num_tensors; i++)
            {
                // Get the ID and create a tensor
                neuropod::SHMBlockID block_id;
                std::copy_n(received.tensor_id[i], block_id.size(), block_id.begin());
                auto shm_tensor = tensor_from_id(block_id);

                // Make sure we're not overwriting a tensor
                std::string tensor_name = received.tensor_name[i];
                assert(inputs.find(tensor_name) == inputs.end());

                // Wrap in a tensor type that this neuropod expects
                inputs[tensor_name] = wrap_existing_tensor(*allocator, shm_tensor);
            }
        }
        else if (received.type == REQUEST_OUTPUT)
        {
            for (int i = 0; i < received.num_tensors; i++)
            {
                std::string tensor_name = received.tensor_name[i];
                requested_outputs.emplace_back(tensor_name);
            }
        }
        else if (received.type == INFER)
        {
            // Run inference
            auto outputs = neuropod->infer(inputs, requested_outputs);

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

            // Clear the requested outputs
            requested_outputs.clear();

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
