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

#include "neuropod/internal/logging.hh"
#include "neuropod/multiprocess/control_messages.hh"
#include "neuropod/multiprocess/ipc_control_channel.hh"
#include "neuropod/multiprocess/ope_load_config.hh"
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

    while (true)
    {
        // Get a message
        auto received = control_channel.recv_message();
        auto msg_type = received.get_payload_type();

        try
        {
            if (msg_type == LOAD_NEUROPOD)
            {
                ope_load_config config;
                received.get(config);

                // Override some options
                auto &opts                      = config.opts;
                opts.load_model_at_construction = true;
                opts.use_ope                    = false;

                // Load a neuropod
                neuropod  = stdx::make_unique<Neuropod>(config.neuropod_path, config.default_backend_overrides, opts);
                allocator = neuropod->get_tensor_allocator();
                inputs.clear();
                control_channel.send_message(LOAD_SUCCESS);
            }
            else if (msg_type == ADD_INPUT)
            {
                NeuropodValueMap tmp;
                received.get(tmp);

                for (auto &item : tmp)
                {
                    // Wrap in a tensor type that this neuropod expects
                    inputs[item.first] =
                        wrap_existing_tensor(*allocator, std::dynamic_pointer_cast<NeuropodTensor>(item.second));
                }
            }
            else if (msg_type == INFER)
            {
                // Get the requested tensor names
                std::vector<std::string> requested_outputs;
                received.get(requested_outputs);

                // Run inference
                auto outputs = neuropod->infer(inputs, requested_outputs);

                // Turn these "native" tensors into shm tensors
                NeuropodValueMap transformed_outputs;
                for (const auto &entry : *outputs)
                {
                    // Unfortunately, this requires a copy (done within SHMNeuropodTensor)
                    auto shm_tensor = wrap_existing_tensor<SHMNeuropodTensor>(
                        std::dynamic_pointer_cast<NeuropodTensor>(entry.second));

                    // This ensures that the tensor stays around long enough for the other process to load it
                    transformed_outputs[entry.first] = shm_tensor;
                }

                control_channel.send_message_move(RETURN_OUTPUT, std::move(transformed_outputs));

                // Clean up any unused shm tensors that haven't been reused
                shm_allocator.free_unused_shm_blocks();

                // Empty the inputs set. This is done after sending outputs back to the main process
                // because this takes a nontrivial amount of time
                inputs.clear();
            }
            else if (msg_type == SHUTDOWN)
            {
                break;
            }
            else
            {
                NEUROPOD_ERROR("OPE: Unhandled message type: {}", msg_type);
            }
        }
        catch (const std::exception &e)
        {
            // Send the exception info back to the main process
            std::string msg = e.what();
            control_channel.send_message(EXCEPTION, msg);
        }
        catch (...)
        {
            control_channel.send_message(EXCEPTION, "An unknown exception occurred during inference");
        }

        SPDLOG_TRACE("OPE: BOTTOM OF WORKER LOOP");
    }
}

} // namespace neuropod
