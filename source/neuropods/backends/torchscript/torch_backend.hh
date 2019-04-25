//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/backends/torchscript/torch_tensor.hh"

namespace neuropods
{


// This backend can execute TorchScript models using the
// native C++ torch API
class TorchNeuropodBackend : public NeuropodBackendWithDefaultAllocator<TorchNeuropodTensor>
{
private:
    // The loaded TorchScript Module
    std::shared_ptr<torch::jit::script::Module> model_;

public:
    TorchNeuropodBackend(const std::string &neuropod_path, std::unique_ptr<ModelConfig> &model_config);

    // Create a TorchNeuropodBackend using the path to a TorchScript model exported using `torch.jit.save`
    TorchNeuropodBackend(const std::string &torchscript_model_path);

    ~TorchNeuropodBackend();

    // Run inference
    std::unique_ptr<ValueMap> infer(const ValueSet &inputs);
};

} // namespace neuropods
