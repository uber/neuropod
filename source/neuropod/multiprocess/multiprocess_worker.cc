//
// Uber, Inc. (c) 2019
//

#include "neuropod/internal/double_buffered_queue.hh"
#include "neuropod/internal/double_buffered_priority_queue.hh"

#include "neuropod/multiprocess/control_messages.hh"
#include "neuropod/multiprocess/ipc_control_channel.hh"
#include "neuropod/multiprocess/shm_tensor.hh"
#include "neuropod/multiprocess/tensor_utils.hh"
#include "neuropod/neuropod.hh"

#include <atomic>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace neuropod
{

namespace
{

// Starts a new thread to send a heartbeat
class HeartbeatController
{
private:
    std::atomic_bool send_heartbeat_;
    std::thread      heartbeat_thread_;

public:
    HeartbeatController(IPCControlChannel &control_channel, size_t interval_ms)
        : send_heartbeat_(true), heartbeat_thread_([this, &control_channel, interval_ms]() {
              // Send a heartbeat every 2 seconds
              while (send_heartbeat_)
              {
                  // send_message is threadsafe so we don't need
                  // synchronization here
                  control_channel.send_message(HEARTBEAT);
                  std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
              }
          })
    {
    }

    ~HeartbeatController()
    {
        // Join the heartbeat thread
        send_heartbeat_ = false;
        heartbeat_thread_.join();
    }
};

struct LoadedItem
{
    neuropod::SHMBlockID           block_id;
    std::shared_ptr<NeuropodValue> tensor;
};

// Responsible for using a worker thread to load tensors given IDs
class LoadController
{
private:
    DoubleBufferedPriorityQueue<SHMBlockID> input_queue_;
    DoubleBufferedQueue<LoadedItem>         output_queue_;

    std::atomic_bool                          enable_;
    std::thread                               worker_thread_;


    // Only accessed on the worker thread
    // TODO(vip): Make this an LRU cache
    std::map<SHMBlockID, std::shared_ptr<NeuropodValue>> speculative_items_;

    // Items that have been loaded
    std::set<SHMBlockID> loaded_items_;

public:
    LoadController(std::unique_ptr<Neuropod> &neuropod)
        : input_queue_(2), enable_(true), worker_thread_([this, &neuropod]{
            while (enable_)
            {
                LoadedItem item;
                size_t priority;

                // Get an item and process it; waits if the queue is empty
                auto success = input_queue_.dequeue(item.block_id, priority);
                if (!success)
                {
                    break;
                }

                if (priority == 0)
                {
                    // A "real" load

                    // Check if we've speculatively loaded this item
                    auto speculative = speculative_items_.find(item.block_id);
                    if (speculative != speculative_items_.end())
                    {
                        // We loaded this tensor earlier!
                        item.tensor = std::move(speculative->second);
                        speculative_items_.erase(speculative);
                    }
                    else
                    {
                        // Create a tensor
                        auto shm_tensor = tensor_from_id(item.block_id);

                        // Wrap in a tensor type that this neuropod expects
                        item.tensor = wrap_existing_tensor(*neuropod, shm_tensor);

                        // Mark that we loaded it for real
                        loaded_items_.emplace(item.block_id);
                    }

                    // Add it to the output queue
                    output_queue_.enqueue(std::move(item));
                }
                else
                {
                    // A speculative load

                    // Check if we've already loaded this tensor "for real"
                    auto has_loaded_it = loaded_items_.find(item.block_id);
                    if (has_loaded_it != loaded_items_.end())
                    {
                        // We already loaded this for real - don't load it again
                        loaded_items_.erase(has_loaded_it);
                        continue;
                    }

                    // Create a tensor
                    auto shm_tensor = tensor_from_id(item.block_id);

                    // Wrap in a tensor type that this neuropod expects
                    item.tensor = wrap_existing_tensor(*neuropod, shm_tensor);

                    // We'll probably be asked to load this tensor later
                    speculative_items_[item.block_id] = std::move(item.tensor);
                }
            }
        })
    {
    }

    ~LoadController()
    {
        // Disable the worker thread and tell the queue to stop blocking
        enable_ = false;
        input_queue_.shutdown();

        worker_thread_.join();
    }

    void enqueue(SHMBlockID item)
    {
        input_queue_.enqueue(std::move(item), 0);
    }

    void enqueue(std::vector<SHMBlockID> &container)
    {
        input_queue_.enqueue(container, 0);
    }

    // These will be loaded when the worker is idle and can be used
    // to prefetch tensors that will be loaded later
    void speculative_enqueue(SHMBlockID item)
    {
        input_queue_.enqueue(std::move(item), 1);
    }

    void speculative_enqueue(std::vector<SHMBlockID> &container)
    {
        input_queue_.enqueue(container, 1);
    }

    void dequeue(LoadedItem &item)
    {
        output_queue_.dequeue(item);
    }
};

} // namespace

// The main loop for a worker that runs a neuropod
void multiprocess_worker_loop(const std::string &control_queue_name)
{
    // Open the control channels
    IPCControlChannel control_channel(control_queue_name, WORKER_PROCESS);

    // A pointer to a neuropod (that will be loaded)
    std::unique_ptr<Neuropod> neuropod;

    // A map to store the inputs
    NeuropodValueMap inputs;

    // The last outputs
    // We need to keep these around so there isn't a race condition when returning
    // data back to the main process
    NeuropodValueMap last_outputs;

    // Send a heartbeat periodically
    HeartbeatController heartbeat_controller(control_channel, HEARTBEAT_INTERVAL_MS);


    LoadController load_controller_(neuropod);

    std::map<SHMBlockID, std::string> id_to_name_map_;

    while (true)
    {
        // Get a message
        control_message received;
        control_channel.recv_message(received);

        if (received.type == LOAD_NEUROPOD)
        {
            // Load a neuropod
            neuropod = stdx::make_unique<Neuropod>(received.neuropod_path);
            inputs.clear();
            last_outputs.clear();
        }
        else if (received.type == ADD_INPUT)
        {
            std::vector<SHMBlockID> block_ids(received.num_tensors);
            for (int i = 0; i < received.num_tensors; i++)
            {
                // Get the ID and create a tensor
                neuropod::SHMBlockID &block_id = block_ids[i];
                std::copy_n(received.tensor_id[i], block_id.size(), block_id.begin());

                std::string tensor_name = received.tensor_name[i];
                id_to_name_map_[block_id] = tensor_name;
            }

            load_controller_.enqueue(block_ids);
        }
        else if (received.type == SPECULATIVE)
        {
            // Speculatively load tensors
            std::vector<SHMBlockID> block_ids(received.num_tensors);
            for (int i = 0; i < received.num_tensors; i++)
            {
                // Get the ID and create a tensor
                neuropod::SHMBlockID &block_id = block_ids[i];
                std::copy_n(received.tensor_id[i], block_id.size(), block_id.begin());
            }

            load_controller_.speculative_enqueue(block_ids);
        }
        else if (received.type == INFER)
        {
            while (!id_to_name_map_.empty())
            {
                LoadedItem item;
                load_controller_.dequeue(item);

                // Make sure we're not overwriting a tensor
                std::string tensor_name = id_to_name_map_.at(item.block_id);
                id_to_name_map_.erase(item.block_id);
                assert(inputs.find(tensor_name) == inputs.end());

                // Wrap in a tensor type that this neuropod expects
                inputs[tensor_name] = item.tensor;
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
