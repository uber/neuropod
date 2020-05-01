//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/backends/torchscript/torch_tensor.hh"
#include "neuropod/internal/tensor_types.hh"
#include "neuropod/neuropod.hh"

#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace neuropod
{

// This backend can execute TorchScript models using the
// native C++ torch API
class TorchNeuropodBackend : public NeuropodBackendWithDefaultAllocator<TorchNeuropodTensor>
{
private:
    // The loaded TorchScript Module
    std::shared_ptr<torch::jit::script::Module> model_;

    // The model output specification from ModelConfig
    std::vector<TensorSpec> output_specs_;

    // The device mapping for the input tensors
    std::unordered_map<std::string, NeuropodDeviceType> input_device_mapping_;

    // Get a torch device given a target neuropod device
    // (this also depends on the visible device in the options above)
    torch::Device get_torch_device(NeuropodDeviceType target_device);

public:
    TorchNeuropodBackend(const std::string &neuropod_path, const RuntimeOptions &options);

    ~TorchNeuropodBackend();

protected:
    // Run inference
    std::unique_ptr<NeuropodValueMap> infer_internal(const NeuropodValueMap &inputs);

    // A method that loads the underlying model
    void load_model_internal();
};

} // namespace neuropod
