//
// Uber, Inc. (c) 2019
//

#include <signal.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/backends/multiprocess_backend/shm_tensor.hh"
#include "neuropods/internal/tensor_store.hh"

namespace neuropods
{

namespace
{

// Start a neuropod worker process given a path to a neuropod
void start_worker_process(
    const std::string &neuropod_path,
    pid_t &child_pid,
    int &read_fd,
    int &write_fd
)
{
    // 2 pipes and each has a write and a read
    int pipes[2][2];

    // Make the pipes
    pipe(pipes[0]);
    pipe(pipes[1]);

    // For clarity
    int child_read_fd =   pipes[0][0];
    int parent_write_fd = pipes[0][1];
    int parent_read_fd =  pipes[1][0];
    int child_write_fd =  pipes[1][1];

    child_pid = fork();
    if (child_pid < 0)
    {
        NEUROPOD_ERROR("Failed to start worker process.");
    }
    else if (child_pid == 0)
    {
        // In the child process
        dup2(child_read_fd, STDIN_FILENO);
        dup2(child_write_fd, STDOUT_FILENO);

        // Close all the file descriptors
        close(child_read_fd);
        close(parent_write_fd);
        close(parent_read_fd);
        close(child_write_fd);

        // Start the worker process
        execlp("neuropod_multiprocess_worker", "neuropod_multiprocess_worker", neuropod_path.c_str(), static_cast<char *>(nullptr));
        std::cout << "Failed to start the worker process!" << std::endl;
        exit (EXIT_FAILURE);
    }
    else
    {
        // The parent doesn't need the child file descriptors
        close(child_read_fd);
        close(child_write_fd);

        read_fd = parent_read_fd;
        write_fd = parent_write_fd;
    }
}

} // namespace

namespace io = boost::iostreams;

// This backend lets us run a neuropod in a new process
// It uses shared memory to efficiently send tensors back and forth between the processes
// Inference is zero copy on the input side and requires one copy on the output side
//
// TODO(vip): For some backends, it may be possible to remove this copy as well by forcing
// the underlying DL framework to use a memory allocator we control. This gets quite complicated
// unfortunately.
class MultiprocessNeuropodBackend : public NeuropodBackendWithDefaultAllocator<SHMNeuropodTensor>
{
private:
    // The pid of the child process
    pid_t child_pid_;

    // Used to wrap the file descriptors from the pipes
    std::unique_ptr<io::stream_buffer<io::file_descriptor_source>> ifstream_;
    std::unique_ptr<io::stream_buffer<io::file_descriptor_sink>> ofstream_;

    // Input and output streams used to interact with the worker process
    std::unique_ptr<std::istream> in_;
    std::unique_ptr<std::ostream> out_;

public:
    // Start a new process with the Neuropod path
    // TODO(vip): pass config info through to the worker
    MultiprocessNeuropodBackend(const std::string &neuropod_path, std::unique_ptr<ModelConfig> &model_config)
    {
        int read_fd, write_fd;
        start_worker_process(neuropod_path, child_pid_, read_fd, write_fd);

        // Setup streams from the file descriptors
        ifstream_ = stdx::make_unique<io::stream_buffer<io::file_descriptor_source>>(read_fd, io::close_handle);
        in_ = stdx::make_unique<std::istream>(ifstream_.get());

        ofstream_ = stdx::make_unique<io::stream_buffer<io::file_descriptor_sink>>(write_fd, io::close_handle);
        out_ = stdx::make_unique<std::ostream>(ofstream_.get());

        // While the worker process is running and we can get input data
        std::string line;
        while (std::getline(*in_, line) && !kill(child_pid_, 0))
        {
            // Wait for the worker process to get ready
            if (line == "ready")
            {
                break;
            }
        }
    }

    ~MultiprocessNeuropodBackend()
    {
        // Kill the child process
        kill(child_pid_, SIGKILL);
    }

    // Run inference
    std::unique_ptr<TensorStore> infer(const std::unordered_set<std::shared_ptr<NeuropodTensor>> &inputs)
    {
        // Get the tensor names and pass them to the worker process
        for (const auto &tensor : inputs)
        {
            const auto &shm_key
                = std::dynamic_pointer_cast<NativeDataContainer<std::string>>(tensor)->get_native_data();

            // Send the shm key to the worker process
            // The worker process owns these tensors now
            *out_ << shm_key << std::endl;
        }

        // Run inference
        *out_ << "infer" << std::endl;

        // Get the output tensor names from the worker process
        auto to_return = stdx::make_unique<TensorStore>();
        std::string line;

        // While the input stream is open and the child process is running
        while (std::getline(*in_, line) && !kill(child_pid_, 0))
        {
            if (line == "end_output")
            {
                break;
            }

            auto tensor = tensor_from_shm_key(line);
            to_return->tensors.emplace_back(tensor);
        }

        // Let the worker know that we loaded the output tensors and it
        // no longer needs to keep references to them
        *out_ << "infer_complete" << std::endl;

        return to_return;
    }
};

REGISTER_NEUROPOD_BACKEND(MultiprocessNeuropodBackend, "multiprocess")

} // namespace neuropods
