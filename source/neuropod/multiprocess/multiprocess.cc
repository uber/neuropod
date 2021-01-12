/* Copyright (c) 2020 UATC, LLC

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

#include "neuropod/multiprocess/multiprocess.hh"

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/internal/cuda_device_mapping.hh"
#include "neuropod/internal/logging.hh"
#include "neuropod/multiprocess/control_messages.hh"
#include "neuropod/multiprocess/ipc_control_channel.hh"
#include "neuropod/multiprocess/ope_load_config.hh"
#include "neuropod/multiprocess/shm_tensor.hh"

#include <boost/date_time/microsec_time_clock.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <sys/wait.h>

#include <csignal>
#include <vector>

#include <spawn.h>

extern char **environ;

namespace neuropod
{

namespace
{

// A utility to get the environment as a map
std::unordered_map<std::string, std::string> get_env_map()
{
    std::unordered_map<std::string, std::string> env;
    for (char **current = environ; *current; current++)
    {
        std::string item = *current;
        const auto  pos  = item.find('=');
        if (pos == std::string::npos)
        {
            // No `=` found
            continue;
        }

        const auto key = item.substr(0, pos);  // Not including the `=`
        const auto val = item.substr(pos + 1); // Not including the `=`

        env[key] = val;
    }

    // Base directory for Neuropod backends
    if (auto base_dir = std::getenv("NEUROPOD_BASE_DIR"))
    {
        env["NEUROPOD_BASE_DIR"] = base_dir;
        SPDLOG_TRACE("set NEUROPOD_BASE_DIR={}", base_dir);
    }

    return env;
}

// Start a neuropod worker process given a control queue name
pid_t start_worker_process(const std::string &control_queue_name, std::vector<std::string> env)
{
    pid_t child_pid;
    char *argv[] = {
        const_cast<char *>("neuropod_multiprocess_worker"), const_cast<char *>(control_queue_name.c_str()), nullptr};

    // Setup the environment

    // Null terminated char * array
    char *env_arr[env.size() + 1];
    env_arr[env.size()] = nullptr;

    // Set the env
    for (int i = 0; i < env.size(); i++)
    {
        env_arr[i] = const_cast<char *>(env[i].c_str());
    }

    // Spawn a process
    const auto status = posix_spawnp(&child_pid, "neuropod_multiprocess_worker", nullptr, nullptr, argv, env_arr);
    if (status != 0)
    {
        NEUROPOD_ERROR("Failed to start the worker process. Failed with code: {} - {}", status, strerror(status));
    }

    return child_pid;
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

    // The load config to send to the worker process
    ope_load_config load_config_;

    // Control channel for interacting with the worker
    IPCControlChannel control_channel_;

    void wait_for_load_confirmation(const std::string &neuropod_path)
    {
        // Wait for confirmation that the model was loaded
        SPDLOG_DEBUG("OPE: Waiting for load confirmation from worker...");
        auto received = control_channel_.recv_message();
        auto msg_type = received.get_payload_type();

        if (msg_type == EXCEPTION)
        {
            // Get the message
            std::string msg;
            received.get(msg);

            NEUROPOD_ERROR("Got an exception when loading the model at {}: {}", neuropod_path, msg);
        }

        if (msg_type != LOAD_SUCCESS)
        {
            // We got an unexpected message
            NEUROPOD_ERROR("Expected LOAD_SUCCESS, but got unexpected message from the worker process: {}", msg_type);
        }
    }

public:
    MultiprocessNeuropodBackend(const std::string &neuropod_path,
                                const std::string &control_queue_name,
                                bool               free_memory_every_cycle)
        : NeuropodBackendWithDefaultAllocator<SHMNeuropodTensor>(neuropod_path, {}),
          control_queue_name_(control_queue_name),
          free_memory_every_cycle_(free_memory_every_cycle),
          control_channel_(control_queue_name, MAIN_PROCESS)
    {
        // Setup the load configuration
        load_config_.neuropod_path = neuropod_path_;

        // Load the model
        load_model();
    }

    // Generate a control queue name and start a worker
    MultiprocessNeuropodBackend(const std::string &                 neuropod_path,
                                const RuntimeOptions &              options,
                                bool                                free_memory_every_cycle,
                                const std::vector<BackendLoadSpec> &default_backend_overrides)
        : NeuropodBackendWithDefaultAllocator<SHMNeuropodTensor>(neuropod_path, options),
          control_queue_name_(boost::uuids::to_string(boost::uuids::random_generator()())),
          free_memory_every_cycle_(free_memory_every_cycle),
          control_channel_(control_queue_name_, MAIN_PROCESS)
    {
        auto env = get_env_map();

        // Set the visible devices correctly when starting the worker process
        if (options.visible_device == Device::CPU)
        {
            env["CUDA_VISIBLE_DEVICES"] = "";
        }
        else
        {
            // The GPU UUID is a standard id that is not affected by CUDA_VISIBLE_DEVICES so we can
            // use it to have stable IDs across processes (e.g. for OPE)
            env["CUDA_VISIBLE_DEVICES"] = get_gpu_uuid(options.visible_device);
        }

        // Convert to a vector
        std::vector<std::string> env_vec;
        env_vec.reserve(env.size());
        for (const auto &item : env)
        {
            env_vec.emplace_back(item.first + "=" + item.second);
        }

        // Start the worker process
        child_pid_ = start_worker_process(control_queue_name_, env_vec);

        // Setup the load configuration
        load_config_.neuropod_path             = neuropod_path_;
        load_config_.default_backend_overrides = default_backend_overrides;

        // Copy options into the load configuration
        // Note: some of these options will be overridden in the worker process
        load_config_.opts = options_;

        // Since we're using CUDA_VISIBLE_DEVICES to set the appropriate device above,
        // we'll just tell the worker to use GPU0
        load_config_.opts.visible_device = Device::GPU0;

        if (options.load_model_at_construction)
        {
            load_model();
        }
    }

    ~MultiprocessNeuropodBackend() override
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

protected:
    // Run inference
    std::unique_ptr<NeuropodValueMap> infer_internal(const NeuropodValueMap &        inputs,
                                                     const std::vector<std::string> &requested_outputs) override
    {
        // Add inputs
        control_channel_.send_message_move(ADD_INPUT, std::move(inputs));

        // Run inference with a set of requested outputs
        control_channel_.send_message(INFER, requested_outputs);

        // Get the outputs from the worker
        auto received = control_channel_.recv_message();
        auto msg_type = received.get_payload_type();

        if (msg_type == EXCEPTION)
        {
            // Get the message
            std::string msg;
            received.get(msg);

            NEUROPOD_ERROR("Got an exception during inference: {}", msg);
        }

        if (msg_type != RETURN_OUTPUT)
        {
            NEUROPOD_ERROR("Got unexpected message from the worker process: {}", msg_type);
        }

        // Load the returned tensors
        auto to_return = stdx::make_unique<NeuropodValueMap>();
        received.get(*to_return);

        if (free_memory_every_cycle_)
        {
            // Clean up any unused shm tensors that haven't been reused
            shm_allocator.free_unused_shm_blocks();
        }

        return to_return;
    }

    void load_model_internal() override
    {
        // Send a message to load the model
        control_channel_.send_message(LOAD_NEUROPOD, load_config_);

        // Wait until the worker process confirms it has loaded the model
        wait_for_load_confirmation(neuropod_path_);
    }
};

} // namespace

std::unique_ptr<NeuropodBackend> load_neuropod_ope(const std::string &                 neuropod_path,
                                                   const RuntimeOptions &              options,
                                                   const std::vector<BackendLoadSpec> &default_backend_overrides)
{
    if (!options.use_ope)
    {
        NEUROPOD_ERROR("`load_neuropod_ope` was called, but `options.use_ope` was false");
    }

    const auto  free_memory_every_cycle = options.ope_options.free_memory_every_cycle;
    const auto &control_queue_name      = options.ope_options.control_queue_name;
    if (control_queue_name.empty())
    {
        // Start a new worker
        return stdx::make_unique<MultiprocessNeuropodBackend>(
            neuropod_path, options, free_memory_every_cycle, default_backend_overrides);
    }

    if (!default_backend_overrides.empty())
    {
        // For now, we can't provide overrides and use an existing worker
        NEUROPOD_ERROR("`default_backend_overrides` cannot be specified when using an existing worker (i.e. when "
                       "`control_queue_name` is not empty)");
    }

    // Use an existing worker
    return stdx::make_unique<MultiprocessNeuropodBackend>(neuropod_path, control_queue_name, free_memory_every_cycle);
}

} // namespace neuropod
