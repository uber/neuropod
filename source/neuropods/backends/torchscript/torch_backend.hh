//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/backends/torchscript/torch_tensor.hh"
#include "neuropods/internal/tensor_types.hh"
#include "neuropods/neuropods.hh"

#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace neuropods
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

    // The options this model was loaded with
    RuntimeOptions options_;

    // The device mapping for the input tensors
    std::unordered_map<std::string, DeviceType> input_device_mapping_;

    // Get a torch device given a target neuropod device
    // (this also depends on the visible device in the options above)
    torch::Device get_torch_device(neuropods::DeviceType target_device);

public:
    TorchNeuropodBackend(const std::string &           neuropod_path,
                         std::unique_ptr<ModelConfig> &model_config,
                         const RuntimeOptions &        options);

    // Create a TorchNeuropodBackend using the path to a TorchScript model exported using `torch.jit.save`
    TorchNeuropodBackend(const std::string &torchscript_model_path);

    // Create a TorchNeuropodBackend using the path to a TorchScript model exported using `torch.jit.save` along
    // with a list of custom ops to load
    TorchNeuropodBackend(const std::string &torchscript_model_path, const std::vector<std::string> &custom_op_paths);

    ~TorchNeuropodBackend();

    // Run inference
    std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &inputs);
};

} // namespace neuropods
