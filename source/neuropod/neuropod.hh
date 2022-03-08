/* Copyright (c) 2020 The Neuropod Authors

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

#pragma once

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/internal/config_utils.hh"
#include "neuropod/options.hh"
#include "neuropod/version.hh"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuropod
{

class Neuropod
{
private:
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
    Neuropod(const std::string &                 neuropod_path,
             const std::vector<BackendLoadSpec> &default_backend_overrides,
             const RuntimeOptions &              options = {});

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
    std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &        inputs,
                                            const std::vector<std::string> &requested_outputs = {});

    // If `load_model_at_construction` is false in the RuntimeOptions passed into the constructor,
    // this method loads the model
    void load_model();

    // Get the inputs and outputs of the loaded Neuropod
    const std::vector<TensorSpec> &get_inputs() const;
    const std::vector<TensorSpec> &get_outputs() const;

    // Get the name of the loaded Neuropod.
    const std::string &get_name() const;
    // Get the platform of the loaded Neuropod.
    const std::string &get_platform() const;

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

} // namespace neuropod
