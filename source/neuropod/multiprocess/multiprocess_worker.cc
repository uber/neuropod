//
// Uber, Inc. (c) 2019
//

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

#include "tbb/tbb.h"
#include "tbb/task_group.h"

namespace neuropod
{

// The main loop for a worker that runs a neuropod
void multiprocess_worker_loop(const std::string &control_queue_name)
{
    // Use a single thread
    tbb::task_scheduler_init init(1);

    // Open the control channels
    auto control_channel = std::make_shared<IPCControlChannel>(control_queue_name, WORKER_PROCESS);

    // A pointer to a neuropod (that will be loaded)
    std::unique_ptr<Neuropod>                neuropod;
    std::shared_ptr<NeuropodTensorAllocator> allocator;
    std::unique_ptr<Sealer>                  sealer;
    std::unique_ptr<tbb::task_group>         seal_group;

    // A map to store items that have been sealed
    // TODO(vip): Switch to unordered map once SHMBlockID is hashable
    std::map<SHMBlockID, std::shared_ptr<NeuropodValue>> sealed;
    std::mutex sealed_mutex;

    // A map to store the inputs
    NeuropodValueMap inputs;

    while (true)
    {
        // Get a message
        auto received = control_channel->recv_message();
        auto msg_type = received.get_payload_type();

        try
        {
            if (msg_type == LOAD_NEUROPOD)
            {
                ope_load_config config;
                received.get(config);

                // Load a neuropod
                neuropod   = stdx::make_unique<Neuropod>(config.neuropod_path, config.default_backend_overrides);
                allocator  = neuropod->get_tensor_allocator();
                sealer     = stdx::make_unique<Sealer>(neuropod->get_sealer());
                seal_group = stdx::make_unique<tbb::task_group>();

                sealed.clear();
                inputs.clear();
                control_channel->send_message(LOAD_SUCCESS);
            }
            else if (msg_type == SEAL)
            {
                auto items = std::make_shared<std::vector<SealedSHMTensor>>();
                received.get(*items);

                // Kick off a task to seal the tensors
                seal_group->run([items, allocator, &sealed, &sealed_mutex] {
                    std::map<SHMBlockID, std::shared_ptr<NeuropodValue>> tmp;
                    for (const auto &item : *items)
                    {
                        auto tensor = tensor_from_id(item.block_id);

                        // Wrap in a tensor type that this neuropod expects
                        auto wrapped =
                            wrap_existing_tensor(*allocator, std::dynamic_pointer_cast<NeuropodTensor>(tensor));

                        tmp[item.block_id] = wrapped->seal(item.device);
                    }

                    // Lock the mutex and add inputs
                    std::lock_guard<std::mutex> lock(sealed_mutex);
                    for (auto &item : tmp)
                    {
                        sealed.insert(std::move(item));
                    }
                });
            }
            else if (msg_type == ADD_INPUT)
            {
                // Wait on the task group and then reset it
                seal_group->wait();
                seal_group = stdx::make_unique<tbb::task_group>();

                // Grab the sealed tensors
                std::unordered_map<std::string, SHMBlockID> tmp;
                received.get(tmp);

                for (auto &item : tmp)
                {
                    // Get it from the sealed map (since all inputs are sealed before inference)
                    inputs[item.first] = std::move(sealed.at(item.second));
                    sealed.erase(item.second);
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

                control_channel->send_message_move(RETURN_OUTPUT, std::move(transformed_outputs));

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
            control_channel->send_message(EXCEPTION, msg);
        }
        catch (...)
        {
            control_channel->send_message(EXCEPTION, "An unknown exception occurred during inference");
        }

        SPDLOG_TRACE("OPE: BOTTOM OF WORKER LOOP");
    }
}

} // namespace neuropod
