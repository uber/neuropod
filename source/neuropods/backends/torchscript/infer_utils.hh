//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/neuropods.hh"

#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace neuropods
{

class TorchInferenceWrapper
{
private:
    // The loaded TorchScript Module
    std::shared_ptr<torch::jit::script::Module> model_;

public:
    // Load a TorchScript model exported using `torch.jit.save`.
    // Allows the user to specify optional custom op paths and a target device to load the model on.
    TorchInferenceWrapper(const std::string &             torchscript_model_path,
                          const std::vector<std::string> &custom_op_paths = {},
                          const torch::Device &           device = torch::kCUDA);

    ~TorchInferenceWrapper();

    // Run inference
    // If `remap_dict_input` is `true` and the model has a single dict input, pass in all the
    // `inputs` as a single dictionary.
    //
    // Note: the inputs must be on the expected devices before inference
    torch::jit::IValue infer(const std::unordered_map<std::string, torch::jit::IValue> &inputs, bool remap_dict_input);
};

} // namespace neuropods
