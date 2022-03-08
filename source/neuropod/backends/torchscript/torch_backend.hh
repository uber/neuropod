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
