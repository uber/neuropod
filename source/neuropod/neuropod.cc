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

#include "neuropod.hh"

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/internal/backend_registration.hh"
#include "neuropod/internal/config_utils.hh"
#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/neuropod_tensor.hh"
#include "neuropod/multiprocess/multiprocess.hh"

namespace neuropod
{

Neuropod::Neuropod(const std::string &neuropod_path, const RuntimeOptions &options)
    : Neuropod(neuropod_path, {}, options)
{
}

// Find the right backend to use and load the neuropod
Neuropod::Neuropod(const std::string &                 neuropod_path,
                   const std::vector<BackendLoadSpec> &default_backend_overrides,
                   const RuntimeOptions &              options)
{
    if (options.use_ope)
    {
        // Load the model using OPE
        backend_ = load_neuropod_ope(neuropod_path, options, default_backend_overrides);
    }
    else
    {
        // Get the backend from the registered backends
        const auto model_config = load_model_config(neuropod_path);
        backend_                = get_backend_for_type(default_backend_overrides,
                                        model_config->platform,
                                        model_config->platform_version_semver)(neuropod_path, options);
    }
}

// Load the model config and use the backend that was provided by the user
Neuropod::Neuropod(const std::string &neuropod_path, std::shared_ptr<NeuropodBackend> backend)
    : backend_(std::move(backend))
{
}

Neuropod::~Neuropod() = default;

void Neuropod::load_model()
{
    backend_->load_model();
}

std::unique_ptr<NeuropodValueMap> Neuropod::infer(const NeuropodValueMap &        inputs,
                                                  const std::vector<std::string> &requested_outputs)
{
    // TODO(vip): make sure that names in `inputs` are not repeated
    // Run inference
    return backend_->infer(inputs, requested_outputs);
}

const std::vector<TensorSpec> &Neuropod::get_inputs() const
{
    return backend_->get_inputs();
}

const std::vector<TensorSpec> &Neuropod::get_outputs() const
{
    return backend_->get_outputs();
}

const std::string &Neuropod::get_name() const
{
    return backend_->get_name();
}

const std::string &Neuropod::get_platform() const
{
    return backend_->get_platform();
}

std::shared_ptr<NeuropodTensorAllocator> Neuropod::get_tensor_allocator()
{
    return backend_->get_tensor_allocator();
}

template <typename T>
std::shared_ptr<TypedNeuropodTensor<T>> Neuropod::allocate_tensor(const std::vector<int64_t> &input_dims)
{
    return get_tensor_allocator()->allocate_tensor<T>(input_dims);
}

template <typename T>
std::shared_ptr<TypedNeuropodTensor<T>> Neuropod::tensor_from_memory(const std::vector<int64_t> &input_dims,
                                                                     T *                         data,
                                                                     const Deleter &             deleter)
{
    return get_tensor_allocator()->tensor_from_memory(input_dims, data, deleter);
}

// Instantiate the templates
#define INIT_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE)                                  \
    template std::shared_ptr<TypedNeuropodTensor<CPP_TYPE>> Neuropod::tensor_from_memory( \
        const std::vector<int64_t> &input_dims, CPP_TYPE *data, const Deleter &deleter);

#define INIT_STRING_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE)                        \
    template std::shared_ptr<TypedNeuropodTensor<CPP_TYPE>> Neuropod::allocate_tensor( \
        const std::vector<int64_t> &input_dims);

FOR_EACH_TYPE_MAPPING_EXCEPT_STRING(INIT_TEMPLATES_FOR_TYPE);
FOR_EACH_TYPE_MAPPING_INCLUDING_STRING(INIT_STRING_TEMPLATES_FOR_TYPE);

} // namespace neuropod
