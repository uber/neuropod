//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/internal/config_utils.hh"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuropods
{

typedef int NeuropodDevice;
namespace Device
{
    constexpr int CPU  = -1;
    constexpr int GPU0 = 0;
    constexpr int GPU1 = 1;
    constexpr int GPU2 = 2;
    constexpr int GPU3 = 3;
    constexpr int GPU4 = 4;
    constexpr int GPU5 = 5;
    constexpr int GPU6 = 6;
    constexpr int GPU7 = 7;
}

struct RuntimeOptions
{
    // The device to run this Neuropod on.
    // Some devices are defined in the namespace above. For machines with more
    // than 8 GPUs, passing in an index will also work (e.g. `9` for `GPU9`).
    //
    // To attempt to run the model on CPU, set this to `Device::CPU`
    NeuropodDevice visible_device = Device::GPU0;
};

class Neuropod
{
private:
    // The neuropod model config
    std::unique_ptr<ModelConfig> model_config_;

    // The backend used to load and run the neuropod
    std::shared_ptr<NeuropodBackend> backend_;

public:
    // Load a neuropod.
    Neuropod(const std::string &neuropod_path, const RuntimeOptions &options = {});

    // Load a neuropod with custom default backends.
    // `default_backend_overrides` allows users to override the default backend for a given type.
    // This is a mapping from a neuropod type (e.g. tensorflow, python, torchscript, etc.) to the
    // name of a shared library that supports that type.
    // Note: Libraries in this map will only be loaded if a backend for the requested type hasn't
    // already been loaded
    Neuropod(const std::string &                                 neuropod_path,
             const std::unordered_map<std::string, std::string> &default_backend_overrides,
             const RuntimeOptions &options = {});

    // Use a specific backend to execute the neuropod
    Neuropod(const std::string &neuropod_path,
             const std::string &backend_name,
             const RuntimeOptions &options = {});

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
    std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &inputs);

    // Get the inputs and outputs of the loaded Neuropod
    const std::vector<TensorSpec> &get_inputs() const;
    const std::vector<TensorSpec> &get_outputs() const;

    // Returns a tensor allocator that can allocate tensors compatible with this neuropod
    std::shared_ptr<NeuropodTensorAllocator> get_tensor_allocator();

    // Allocate a tensor of a certain shape and type
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> allocate_tensor(const std::vector<int64_t> &input_dims);

    // Allocate a tensor of a certain shape and type with existing data
    // The user-provided memory will be wrapped and `deleter`
    // will be called with a pointer to `data` when the tensor is
    // deallocated.
    //
    // Note: Some backends may have specific alignment requirements (e.g. tensorflow).
    // To support all the built-in backends, `data` should be aligned to 64 bytes.
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> tensor_from_memory(const std::vector<int64_t> &input_dims,
                                                               T *                         data,
                                                               const Deleter &             deleter);
};

} // namespace neuropods
