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
                neuropods::send_message(queue, HEARTBEAT);
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

// A worker process that runs a neuropod
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::string program_name(argv[0]);
        std::cout << "Usage: " + program_name + " control_queue_name" << std::endl;
        return 1;
    }

    // Open the control channels
    std::string control_queue_name(argv[1]);
    ipc::message_queue to_worker(ipc::open_or_create, ("neuropod_" + control_queue_name + "_tw").c_str(), 20, sizeof(control_message));
    ipc::message_queue from_worker(ipc::open_or_create, ("neuropod_" + control_queue_name + "_fw").c_str(), 20, sizeof(control_message));

    // A pointer to a neuropod (that will be loaded)
    std::unique_ptr<neuropods::Neuropod> neuropod;

    // A map to store the inputs
    neuropods::NeuropodValueMap inputs;

    // The last outputs
    // We need to keep these around so there isn't a race condition when returning
    // data back to the main process
    neuropods::NeuropodValueMap last_outputs;

    // Send a heartbeat on the `from_worker` queue every 2 seconds
    HeartbeatController heartbeat_controller(from_worker, 2000);

    while (true)
    {
        // Get a message
        control_message received;
        size_t received_size;
        unsigned int priority;
        to_worker.receive(&received, sizeof(control_message), received_size, priority);

        if (received.type == LOAD_NEUROPOD)
        {
            // Load a neuropod
            neuropod = neuropods::stdx::make_unique<neuropods::Neuropod>(received.neuropod_path);
            inputs.clear();
            last_outputs.clear();
        }
        else if (received.type == ADD_INPUT)
        {
            for (int i = 0; i < received.num_tensors; i++)
            {
                // Get the uuid and create a tensor
                boost::uuids::uuid uuid;
                std::copy_n(received.tensor_uuid[i], uuid.size(), uuid.begin());
                auto shm_tensor = neuropods::tensor_from_uuid(uuid);

                // Wrap in a tensor type that this neuropod expects
                inputs[received.tensor_name[i]] = neuropods::wrap_existing_tensor(*neuropod, shm_tensor);
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
                auto shm_tensor = neuropods::wrap_existing_tensor<neuropods::SHMNeuropodTensor>(std::dynamic_pointer_cast<neuropods::NeuropodTensor>(entry.second));

                // This ensures that the tensor stays around long enough for the other process to load it
                last_outputs[entry.first] = shm_tensor;
            }

            neuropods::send_message(from_worker, RETURN_OUTPUT, last_outputs);

            // Let the main process know that we're done
            neuropods::send_message(from_worker, END_OUTPUT);

            // Clean up any unused shm tensors that haven't been reused
            neuropods::free_unused_shm_blocks();

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
    }
}
