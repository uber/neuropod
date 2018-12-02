//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <vector>
#include <memory>

#include "neuropods/proxy/neuropod_proxy.hh"
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

    // Allows a proxy to be passed in if we want to run the backends
    // on a remote machine or in a different process
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
    // Example usage:
    //   auto proxy = std::make_shared<NeuropodGRPCProxy>(neuropod_path, ...);
    //   Neuropod neuropod(proxy);
    //
    // Note: the NeuropodProxy that is passed in below is already initialized with a path.
    // Therefore, we don't need the user to pass in a path here.
    explicit Neuropod(std::shared_ptr<NeuropodProxy> backend_proxy);

    ~Neuropod();

    // Get a NeuropodInputBuilder
    std::unique_ptr<NeuropodInputBuilder> get_input_builder();

    // Run inference
    // You should use a `NeuropodInputBuilder` to generate the input
    std::unique_ptr<NeuropodOutputData> infer(
        const std::unique_ptr<NeuropodInputData, NeuropodInputDataDeleter> &inputs);
};

} // namespace neuropods
