//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <vector>
#include <memory>

#include "neuropods/neuropod_input_builder.hh"
#include "neuropods/neuropod_output_data.hh"

namespace neuropods
{

class Neuropod
{
private:
    struct impl;
    std::unique_ptr<impl> pimpl;

public:
    // Load a neuropod
    explicit Neuropod(const std::string &neuropod_path);

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
    // Note: the backend that is passed in is already initialized with a path.
    // Therefore, we don't need the user to pass in a path here.
    explicit Neuropod(std::shared_ptr<NeuropodBackend> backend);

    ~Neuropod();

    // Get a NeuropodInputBuilder
    std::unique_ptr<NeuropodInputBuilder> get_input_builder();

    // Run inference
    // You should use a `NeuropodInputBuilder` to generate the input
    std::unique_ptr<NeuropodOutputData> infer(
        const std::unique_ptr<NeuropodInputData, NeuropodInputDataDeleter> &inputs);
};

} // namespace neuropods
