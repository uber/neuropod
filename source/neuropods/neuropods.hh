//
// Uber, Inc. (c) 2018
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "neuropods/internal/config_utils.hh"
#include "neuropods/neuropod_input_builder.hh"

namespace neuropods
{

class Neuropod
{
private:
    struct impl;
    std::unique_ptr<impl> pimpl;

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

    // Get a NeuropodInputBuilder
    std::unique_ptr<NeuropodInputBuilder> get_input_builder();

    // Run inference
    // You should use a `NeuropodInputBuilder` to generate the input
    std::unique_ptr<TensorStore> infer(const std::unique_ptr<TensorStore> &inputs);

    // Get the inputs and outputs of the loaded Neuropod
    const std::vector<TensorSpec> &get_inputs() const;
    const std::vector<TensorSpec> &get_outputs() const;
};

} // namespace neuropods
