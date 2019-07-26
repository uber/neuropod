//
// Uber, Inc. (c) 2019
//

#include "neuropods/multiprocess/multiprocess.hh"

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/multiprocess/control_messages.hh"
#include "neuropods/multiprocess/message_utils.hh"
#include "neuropods/multiprocess/shm_tensor.hh"

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/date_time/microsec_time_clock.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <signal.h>
#include <sys/wait.h>
#include <vector>

namespace ipc = boost::interprocess;

namespace neuropods
{

namespace
{

// Start a neuropod worker process given a control queue name
pid_t start_worker_process(const std::string &control_queue_name)
{
    pid_t child_pid = fork();
    if (child_pid < 0)
    {
        NEUROPOD_ERROR("Failed to start worker process.");
    }
    else if (child_pid == 0)
    {
        // In the child process
        // Start the worker
        execlp("neuropod_multiprocess_worker", "neuropod_multiprocess_worker", control_queue_name.c_str(), static_cast<char *>(nullptr));

        // If we get here, execlp failed
        std::cerr << "Failed to start the worker process. Failed with code: " << errno << ": " << strerror(errno) << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        // In the parent process
        return child_pid;
    }
}

// Note: we don't register this with the library as a backend because it is not
// a backend in the normal sense. It is only used here for out of process
// execution

class MultiprocessNeuropodBackend : public NeuropodBackendWithDefaultAllocator<SHMNeuropodTensor>
{
private:
    pid_t child_pid_ = -1;
    std::string control_queue_name_;
    bool free_memory_every_cycle_;

    // Control channels for interacting with the worker
    std::unique_ptr<ipc::message_queue> to_worker_;
    std::unique_ptr<ipc::message_queue> from_worker_;
public:
    MultiprocessNeuropodBackend(const std::string &neuropod_path, const std::string &control_queue_name, bool free_memory_every_cycle)
        : control_queue_name_(control_queue_name), free_memory_every_cycle_(free_memory_every_cycle)
    {
        // Create the control channels
        to_worker_ = stdx::make_unique<ipc::message_queue>(ipc::open_or_create, ("neuropod_" + control_queue_name_ + "_tw").c_str(), 20, sizeof(control_message));
        from_worker_ = stdx::make_unique<ipc::message_queue>(ipc::open_or_create, ("neuropod_" + control_queue_name_ + "_fw").c_str(), 20, sizeof(control_message));

        // Create a message to tell the worker process to load the specified neuropod
        control_message msg;
        msg.type = LOAD_NEUROPOD;
        if (neuropod_path.size() >= 4096)
        {
            NEUROPOD_ERROR("The multiprocess backend only supports neuropod paths < 4096 characters long.")
        }

        // Copy in the path
        strncpy(msg.neuropod_path, neuropod_path.c_str(), 4096);

        // Send the message
        to_worker_->send(&msg, sizeof(control_message), 0);
    }

    // Generate a control queue name and start a worker
    MultiprocessNeuropodBackend(const std::string &neuropod_path, bool free_memory_every_cycle) : MultiprocessNeuropodBackend(neuropod_path, boost::uuids::to_string(boost::uuids::random_generator()()), free_memory_every_cycle)
    {
        // Start the worker process
        child_pid_ = start_worker_process(control_queue_name_);
    }

    ~MultiprocessNeuropodBackend()
    {
        // We only need to clean up all of this if we started the worker process
        if (child_pid_ > 0)
        {
            // Kill the child process
            kill(child_pid_, SIGKILL);

            // Wait for it
            // TODO(vip): Check the output of waitpid
            int status;
            waitpid(child_pid_, &status, 0);

            // Delete the control channels
            ipc::message_queue::remove(("neuropod_" + control_queue_name_ + "_tw").c_str());
            ipc::message_queue::remove(("neuropod_" + control_queue_name_ + "_fw").c_str());
        }
    }

    // Run inference
    std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &inputs)
    {
        if (free_memory_every_cycle_)
        {
            // Clean up any unused shm tensors that haven't been reused
            free_unused_shm_blocks();
        }

        // Add inputs
        send_message(*to_worker_, ADD_INPUT, inputs);

        // Run inference
        send_message(*to_worker_, INFER);

        // Get the outputs from the worker
        auto to_return = stdx::make_unique<NeuropodValueMap>();
        while (true) {
            // Get a message from the worker
            control_message received;
            size_t received_size;
            unsigned int priority;

            // 5 second timeout
            // This is generous since the worker sends heartbeats every 2 seconds
            auto timeout_at = boost::posix_time::microsec_clock::local_time() + boost::posix_time::seconds(5);

            bool successful_read = from_worker_->timed_receive(&received, sizeof(control_message), received_size, priority, timeout_at);
            if (!successful_read)
            {
                // We timed out
                NEUROPOD_ERROR("Timed out waiting for response from worker process");
            }

            if (received.type == END_OUTPUT)
            {
                // Got all the outputs
                break;
            }

            if (received.type == HEARTBEAT)
            {
                // TODO(vip): Also periodically check for a heartbeat
                continue;
            }

            if (received.type != RETURN_OUTPUT)
            {
                NEUROPOD_ERROR("Got unexpected message from the worker process:" << received.type)
            }

            for (int i = 0; i < received.num_tensors; i++)
            {
                // Get the uuid and create a tensor
                boost::uuids::uuid uuid;
                std::copy_n(received.tensor_uuid[i], uuid.size(), uuid.begin());
                (*to_return)[received.tensor_name[i]] = tensor_from_uuid(uuid);
            }
        }

        // Inference is complete
        // Let the worker know it no longer needs to keep references to the output
        // tensors
        send_message(*to_worker_, INFER_COMPLETE);

        return to_return;
    }
};

} // namespace

std::unique_ptr<Neuropod> load_neuropod_in_new_process(const std::string &neuropod_path, bool free_memory_every_cycle)
{
    auto backend = std::make_shared<MultiprocessNeuropodBackend>(neuropod_path, free_memory_every_cycle);
    return stdx::make_unique<Neuropod>(neuropod_path, backend);
}

std::unique_ptr<Neuropod> load_neuropod_in_worker(const std::string &neuropod_path, const std::string &control_queue_name, bool free_memory_every_cycle)
{
    auto backend = std::make_shared<MultiprocessNeuropodBackend>(neuropod_path, control_queue_name, free_memory_every_cycle);
    return stdx::make_unique<Neuropod>(neuropod_path, backend);
}

} // namespace neuropods
