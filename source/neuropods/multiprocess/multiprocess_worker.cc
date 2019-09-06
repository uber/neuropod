//
// Uber, Inc. (c) 2019
//

#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <boost/interprocess/ipc/message_queue.hpp>

#include "neuropods/neuropods.hh"
#include "neuropods/multiprocess/control_messages.hh"
#include "neuropods/multiprocess/message_utils.hh"
#include "neuropods/multiprocess/shm_tensor.hh"
#include "neuropods/multiprocess/tensor_utils.hh"

namespace ipc = boost::interprocess;

namespace neuropods
{

namespace
{

// Starts a new thread to send a heartbeat to a message queue
class HeartbeatController
{
private:
    std::atomic_bool send_heartbeat_;
    std::thread      heartbeat_thread_;

public:
    HeartbeatController(ipc::message_queue &queue, size_t interval_ms)
        : send_heartbeat_(true),
          heartbeat_thread_([this, &queue, interval_ms]()
        {
            // Send a heartbeat every 2 seconds
            while (send_heartbeat_) {
                // Boost message_queue is threadsafe so we don't need
                // synchronization here
                send_message(queue, HEARTBEAT);
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
            }
        })
    {}

    ~HeartbeatController()
    {
        // Join the heartbeat thread
        send_heartbeat_ = false;
        heartbeat_thread_.join();
    }
};

} // namespace

// The main loop for a worker that runs a neuropod
void multiprocess_worker_loop(const std::string &control_queue_name)
{
    // Open the control channels
    constexpr auto MAX_QUEUE_SIZE = 20;
    ipc::message_queue to_worker(ipc::open_or_create, ("neuropod_" + control_queue_name + "_tw").c_str(), MAX_QUEUE_SIZE, sizeof(control_message));
    ipc::message_queue from_worker(ipc::open_or_create, ("neuropod_" + control_queue_name + "_fw").c_str(), MAX_QUEUE_SIZE, sizeof(control_message));

    // A pointer to a neuropod (that will be loaded)
    std::unique_ptr<Neuropod> neuropod;

    // A map to store the inputs
    NeuropodValueMap inputs;

    // The last outputs
    // We need to keep these around so there isn't a race condition when returning
    // data back to the main process
    NeuropodValueMap last_outputs;

    // Send a heartbeat on the `from_worker` queue every 2 seconds
    HeartbeatController heartbeat_controller(from_worker, 2000);

    // Verifies that the state machine is operating as expected
    TransitionVerifier verifier;

    while (true)
    {
        // Get a message
        control_message received;
        size_t received_size;
        unsigned int priority;
        to_worker.receive(&received, sizeof(control_message), received_size, priority);

        // Make sure that it is valid to go from the previous message type to the current one
        verifier.assert_transition_allowed(received.type);

        if (received.type == LOAD_NEUROPOD)
        {
            // Load a neuropod
            neuropod = stdx::make_unique<Neuropod>(received.neuropod_path);
            inputs.clear();
            last_outputs.clear();
        }
        else if (received.type == ADD_INPUT)
        {
            for (int i = 0; i < received.num_tensors; i++)
            {
                // Get the ID and create a tensor
                neuropods::SHMBlockID block_id;
                std::copy_n(received.tensor_id[i], block_id.size(), block_id.begin());
                auto shm_tensor = tensor_from_id(block_id);

                // Make sure we're not overwriting a tensor
                std::string tensor_name = received.tensor_name[i];
                assert(inputs.find(tensor_name) == inputs.end());

                // Wrap in a tensor type that this neuropod expects
                inputs[tensor_name] = wrap_existing_tensor(*neuropod, shm_tensor);
            }
        }
        else if (received.type == INFER)
        {
            // Run inference
            auto outputs = neuropod->infer(inputs);

            // Turn these "native" tensors into shm tensors
            for (const auto &entry : *outputs)
            {
                // Unfortunately, this requires a copy (done within SHMNeuropodTensor)
                auto shm_tensor = wrap_existing_tensor<SHMNeuropodTensor>(std::dynamic_pointer_cast<NeuropodTensor>(entry.second));

                // This ensures that the tensor stays around long enough for the other process to load it
                last_outputs[entry.first] = shm_tensor;
            }

            send_message(from_worker, RETURN_OUTPUT, last_outputs);

            // Let the main process know that we're done
            send_message(from_worker, END_OUTPUT);

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

} // namespace neuropods
