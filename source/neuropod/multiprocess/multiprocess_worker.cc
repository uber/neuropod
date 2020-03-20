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

namespace neuropod
{

namespace
{

// Starts a new thread to send a heartbeat
class HeartbeatController
{
private:
    std::atomic_bool        send_heartbeat_;
    std::condition_variable cv_;
    std::mutex              mutex_;
    std::thread             heartbeat_thread_;

public:
    HeartbeatController(IPCControlChannel &control_channel, size_t interval_ms)
        : send_heartbeat_(true), heartbeat_thread_([this, &control_channel, interval_ms]() {
              // Send a heartbeat every 2 seconds
              while (send_heartbeat_)
              {
                  // Note: we're using `try_send_message` instead of `send_message` because
                  // we don't want to block if the message queue is full.
                  // This is important because it could prevent shutdown and cause the worker
                  // and main process to hang.
                  //
                  // For example:
                  // - The main process loads a neuropod using the multiprocess backend (but does not run inference)
                  // - The (worker -> main process) message queue fills up with HEARTBEAT messages
                  //   (because messages are only read from that queue during inference)
                  //
                  // - This thread blocks on sending the next HEARTBEAT message
                  // - The main process sends a SHUTDOWN message and waits for the worker to shut down
                  // - The destructor of `HeartbeatController` runs and runs `join` on this thread
                  //
                  // This will cause both processes to hang forever because the worker is still blocked on the message
                  // queue and the main process is waiting for this process to shutdown.

                  // try_send_message is threadsafe so we don't need synchronization here
                  if (!control_channel.try_send_message(HEARTBEAT))
                  {
                      SPDLOG_DEBUG("OPE: Message queue full - skipped sending heartbeat");
                  }

                  // This lets us break out of waiting if we're told to shutdown
                  std::unique_lock<std::mutex> lk(mutex_);
                  cv_.wait_for(lk, std::chrono::milliseconds(interval_ms), [&] { return send_heartbeat_ != true; });
              }
          })
    {
    }

    ~HeartbeatController()
    {
        // Join the heartbeat thread
        {
            std::lock_guard<std::mutex> lk(mutex_);
            send_heartbeat_ = false;
        }

        cv_.notify_all();
        heartbeat_thread_.join();
    }
};

} // namespace

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
        // Get a message
        ControlMessage received;
        control_channel.recv_message(received);

        auto msg_type = received.get_type();

        if (msg_type == LOAD_NEUROPOD)
        {
            ope_load_config config;
            received.get_load_config(config);

            // Load a neuropod
            neuropod  = stdx::make_unique<Neuropod>(config.neuropod_path);
            allocator = neuropod->get_tensor_allocator();
            inputs.clear();
            last_outputs.clear();
            control_channel.send_message(LOAD_SUCCESS);
        }
        else if (msg_type == ADD_INPUT)
        {
            NeuropodValueMap tmp;
            received.get_valuemap(tmp);

            for (auto &item : tmp)
            {
                // Wrap in a tensor type that this neuropod expects
                inputs[item.first] =
                    wrap_existing_tensor(*allocator, std::dynamic_pointer_cast<NeuropodTensor>(item.second));
            }
        }
        else if (msg_type == REQUEST_OUTPUT)
        {
            // Get the requested tensor names
            received.get_tensor_names(requested_outputs);
        }
        else if (msg_type == INFER)
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
        else if (msg_type == INFER_COMPLETE)
        {
            // The main process loaded our output tensors
            // We no longer need to maintain references to these tensors
            last_outputs.clear();
        }
        else if (msg_type == SHUTDOWN)
        {
            break;
        }
    }
}

} // namespace neuropod
