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

#include "neuropod/backends/tensor_allocator.hh"
#include "neuropod/internal/backend_registration.hh"
#include "neuropod/internal/deleter.hh"
#include "neuropod/internal/neuropod_loader.hh"
#include "neuropod/internal/neuropod_tensor.hh"
#include "neuropod/internal/tensor_types.hh"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace neuropod
{

class Sealer
{
private:
    // A mapping from tensor name to device
    std::unordered_map<std::string, NeuropodDevice> device_mapping_;

public:
    Sealer(std::unordered_map<std::string, NeuropodDevice> device_mapping);
    ~Sealer();

    std::shared_ptr<NeuropodValue> seal(const std::string &name, const std::shared_ptr<NeuropodValue> &value);

    // Seal every item in the map
    NeuropodValueMap seal(const NeuropodValueMap &inputs);
};

// The interface that every neuropod backend implements
class NeuropodBackend
{
public:
    NeuropodBackend(const std::string &neuropod_path, RuntimeOptions options);
    virtual ~NeuropodBackend();

    // Returns an allocator that can allocate tensors compatible with this backend
    virtual std::shared_ptr<NeuropodTensorAllocator> get_tensor_allocator() = 0;

    // Run inference and get a subset of the outputs
    std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &        inputs,
                                            const std::vector<std::string> &requested_outputs = {});

    // Get the inputs and outputs of this model
    const std::vector<TensorSpec> &get_inputs() const;
    const std::vector<TensorSpec> &get_outputs() const;

    // Get the name of this model.
    const std::string &get_name() const;
    // Get the platform of this model.
    const std::string &get_platform() const;

    // Load the model if it has not already been loaded
    void load_model();

protected:
    // Used to load files in a Neuropod
    std::unique_ptr<NeuropodLoader> loader_;

    // The neuropod model config
    std::unique_ptr<ModelConfig> model_config_;

    // The neuropod path (if one was provided in the constructor)
    std::string neuropod_path_;

    // The options this model was loaded with
    RuntimeOptions options_;

    // Run inference and get a subset of the outputs
    // The default implementation runs inference, gets all the outputs, and then filters the outputs
    // Backends can override this to more efficiently generate only the requested outputs
    virtual std::unique_ptr<NeuropodValueMap> infer_internal(const NeuropodValueMap &        inputs,
                                                             const std::vector<std::string> &requested_outputs);

    // Run inference
    // Backends must provide an implementation of infer_internal (either this signature or the one above)
    virtual std::unique_ptr<NeuropodValueMap> infer_internal(const NeuropodValueMap &inputs);

    // A method that loads the underlying model
    virtual void load_model_internal() = 0;

private:
    // Whether or not the underlying model has already been loaded
    bool is_model_loaded_ = false;

    std::unique_ptr<Sealer> sealer_;
};

template <template <class> class TensorImpl>
class NeuropodBackendWithDefaultAllocator : public NeuropodBackend
{
private:
    std::shared_ptr<NeuropodTensorAllocator> allocator_;

public:
    NeuropodBackendWithDefaultAllocator(const std::string &neuropod_path, const RuntimeOptions &options)
        : NeuropodBackend(neuropod_path, options), allocator_(std::make_shared<DefaultTensorAllocator<TensorImpl>>())
    {
    }

    std::shared_ptr<NeuropodTensorAllocator> get_tensor_allocator() { return allocator_; }
};

// A utility for validating tensors against a vector of specs. Throws an error if validation fails
// Note: This function is exposed in order to properly unit test it and should not be directly used
void validate_tensors_against_specs(const NeuropodValueMap &       tensors,
                                    const std::vector<TensorSpec> &specs,
                                    const std::string &            debug_spec_name = "spec");

} // namespace neuropod
