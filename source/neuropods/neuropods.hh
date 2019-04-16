//
// Uber, Inc. (c) 2018
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/internal/config_utils.hh"

namespace neuropods
{

class Neuropod
{
private:
    // The neuropod model config
    std::unique_ptr<ModelConfig> model_config_;

    // The backend used to load and run the neuropod
    std::shared_ptr<NeuropodBackend> backend_;

public:
    // Load a neuropod.
    explicit Neuropod(const std::string &neuropod_path);

    // Load a neuropod with custom default backends.
    // `default_backend_overrides` allows users to override the default backend for a given type.
    // This is a mapping from a neuropod type (e.g. tensorflow, python, torchscript, etc.) to the
    // name of a shared library that supports that type.
    // Note: Libraries in this map will only be loaded if a backend for the requested type hasn't
    // already been loaded
    Neuropod(const std::string &                                 neuropod_path,
             const std::unordered_map<std::string, std::string> &default_backend_overrides);

    // Use a specific backend to execute the neuropod
    Neuropod(const std::string &neuropod_path, const std::string &backend_name);

    // Allows an already-initialized backend to be passed in. This enables backends that need
    // non-standard arguments. For example, this can be used to build a proxy that runs a
    // Neuropod on a remote machine or in a different process.
    //
    // +--------------------------------+                   +----------------------------------+
    // |                                |                   |                                  |
    // |  +-----------+      +-------+  |  GRPC, IPC, etc.  |  +-----------+      +---------+  |
    // |  | Neuropod  | ---> | Proxy |--| ----------------> |  | Neuropod  | ---> | Backend |  |
    // |  +-----------+      +-------+  |                   |  +-----------+      +---------+  |
    // |                                |                   |                                  |
    // +--------------------------------+                   +----------------------------------+
    //              Caller                                                  Worker
    //
    //
    //
    // Example:
    //   auto proxy = std::make_shared<NeuropodGRPCProxy>(neuropod_path, some_remote_config, ...);
    //   Neuropod neuropod(proxy);
    //
    Neuropod(const std::string &neuropod_path, std::shared_ptr<NeuropodBackend> backend);

    ~Neuropod();

    // Run inference
    std::unique_ptr<TensorMap> infer(const TensorSet &inputs);

    // Get the inputs and outputs of the loaded Neuropod
    const std::vector<TensorSpec> &get_inputs() const;
    const std::vector<TensorSpec> &get_outputs() const;

    // Returns a tensor allocator that can allocate tensors compatible with this neuropod
    std::shared_ptr<NeuropodTensorAllocator> get_tensor_allocator();

    // Allocate a tensor of a certain shape and type
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> allocate_tensor(
        const std::string &node_name,
        const std::vector<int64_t> &input_dims);

    // Allocate a tensor of a certain shape and type with existing data
    // The user-provided memory will be wrapped and `deleter`
    // will be called with a pointer to `data` when the tensor is
    // deallocated.
    //
    // Note: Some backends may have specific alignment requirements (e.g. tensorflow).
    // To support all the built-in backends, `data` should be aligned to 64 bytes.
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> tensor_from_memory(
        const std::string &         node_name,
        const std::vector<int64_t> &input_dims,
        T *                         data,
        const Deleter &             deleter);
};

} // namespace neuropods
