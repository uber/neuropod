//
// Uber, Inc. (c) 2019
//

#include "neuropods/multiprocess/multiprocess.hh"

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/multiprocess/control_messages.hh"
#include "neuropods/multiprocess/ipc_control_channel.hh"
#include "neuropods/multiprocess/shm_tensor.hh"

#include <boost/date_time/microsec_time_clock.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <sys/wait.h>

#include <vector>

#include <signal.h>

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
        execlp("neuropod_multiprocess_worker",
               "neuropod_multiprocess_worker",
               control_queue_name.c_str(),
               static_cast<char *>(nullptr));

        // If we get here, execlp failed
        std::cerr << "Failed to start the worker process. Failed with code: " << errno << ": " << strerror(errno)
                  << std::endl;
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
    pid_t       child_pid_ = -1;
    std::string control_queue_name_;
    bool        free_memory_every_cycle_;

    // Control channel for interacting with the worker
    IPCControlChannel control_channel_;

public:
    MultiprocessNeuropodBackend(const std::string &neuropod_path,
                                const std::string &control_queue_name,
                                bool               free_memory_every_cycle)
        : control_queue_name_(control_queue_name),
          free_memory_every_cycle_(free_memory_every_cycle),
          control_channel_(control_queue_name, MAIN_PROCESS)
    {
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
        control_channel_.send_message(msg);
    }

    // Generate a control queue name and start a worker
    MultiprocessNeuropodBackend(const std::string &neuropod_path, bool free_memory_every_cycle)
        : MultiprocessNeuropodBackend(
              neuropod_path, boost::uuids::to_string(boost::uuids::random_generator()()), free_memory_every_cycle)
    {
        // Start the worker process
        child_pid_ = start_worker_process(control_queue_name_);
    }

    ~MultiprocessNeuropodBackend()
    {
        // We only need to clean up all of this if we started the worker process
        if (child_pid_ > 0)
        {
            // Ask the child process to shutdown
            control_channel_.send_message(SHUTDOWN);

            // Wait for it and make sure it exited properly
            int status;
            waitpid(child_pid_, &status, 0);
            if (WIFEXITED(status))
            {
                const auto exit_code = WEXITSTATUS(status);
                if (exit_code != 0)
                {
                    // We don't want to throw an error in the destructor so we'll just log for now
                    std::cerr << "Worker process exited abnormally. Exit code: " << exit_code << std::endl;
                }
            }
            else if (WIFSIGNALED(status))
            {
                // We don't want to throw an error in the destructor so we'll just log for now
                std::cerr << "Worker process exited abnormally. Was terminated by signal: " << WTERMSIG(status)
                          << std::endl;
            }
            else
            {
                // We don't want to throw an error in the destructor so we'll just log for now
                std::cerr << "Worker process exited abnormally." << std::endl;
            }

            // Delete the control channels
            control_channel_.cleanup();
        }
    }

    // Run inference
    std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &inputs)
    {
        if (free_memory_every_cycle_)
        {
            // Clean up any unused shm tensors that haven't been reused
            shm_allocator.free_unused_shm_blocks();
        }

        // Add inputs
        control_channel_.send_message(ADD_INPUT, inputs);

        // Run inference
        control_channel_.send_message(INFER);

        // Get the outputs from the worker
        auto to_return = stdx::make_unique<NeuropodValueMap>();
        while (true)
        {
            // Get a message from the worker
            control_message received;
            bool            successful_read = control_channel_.recv_message(received, MESSAGE_TIMEOUT_MS);
            if (!successful_read)
            {
                // We timed out
                NEUROPOD_ERROR("Timed out waiting for a response from worker process. "
                               "Didn't receive a message in "
                               << MESSAGE_TIMEOUT_MS << "ms, but expected a heartbeat every " << HEARTBEAT_INTERVAL_MS
                               << "ms.");
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
                // Get the ID and create a tensor
                neuropods::SHMBlockID block_id;
                std::copy_n(received.tensor_id[i], block_id.size(), block_id.begin());
                (*to_return)[received.tensor_name[i]] = tensor_from_id(block_id);
            }
        }

        // Inference is complete
        // Let the worker know it no longer needs to keep references to the output
        // tensors
        control_channel_.send_message(INFER_COMPLETE);

        return to_return;
    }
};

} // namespace

std::unique_ptr<Neuropod> load_neuropod_in_new_process(const std::string &neuropod_path, bool free_memory_every_cycle)
{
    auto backend = std::make_shared<MultiprocessNeuropodBackend>(neuropod_path, free_memory_every_cycle);
    return stdx::make_unique<Neuropod>(neuropod_path, backend);
}

std::unique_ptr<Neuropod> load_neuropod_in_worker(const std::string &neuropod_path,
                                                  const std::string &control_queue_name,
                                                  bool               free_memory_every_cycle)
{
    auto backend =
        std::make_shared<MultiprocessNeuropodBackend>(neuropod_path, control_queue_name, free_memory_every_cycle);
    return stdx::make_unique<Neuropod>(neuropod_path, backend);
}

} // namespace neuropods
