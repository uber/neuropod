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

    ~TorchNeuropodBackend();

    // Run inference
    std::unique_ptr<TensorStore> infer(const TensorStore &inputs);
};

} // namespace neuropods
