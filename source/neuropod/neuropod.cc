//
// Uber, Inc. (c) 2018
//

#include "neuropod.hh"

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/internal/backend_registration.hh"
#include "neuropod/internal/config_utils.hh"
#include "neuropod/internal/neuropod_tensor.hh"

namespace neuropod
{

Neuropod::Neuropod(const std::string &neuropod_path, const RuntimeOptions &options)
    : Neuropod(neuropod_path, std::unordered_map<std::string, std::string>(), options)
{
}

// Find the right backend to use and load the neuropod
Neuropod::Neuropod(const std::string &                                 neuropod_path,
                   const std::unordered_map<std::string, std::string> &default_backend_overrides,
                   const RuntimeOptions &                              options)
    : backend_(get_backend_for_type(default_backend_overrides,
                                    load_model_config(neuropod_path)->platform)(neuropod_path, options))
{
}

// Load the neuropod using the specified backend
Neuropod::Neuropod(const std::string &neuropod_path, const std::string &backend_name, const RuntimeOptions &options)
    : backend_(get_backend_by_name(backend_name)(neuropod_path, options))
{
}

// Load the model config and use the backend that was provided by the user
Neuropod::Neuropod(const std::string &neuropod_path, std::shared_ptr<NeuropodBackend> backend) : backend_(backend) {}

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

const std::vector<TensorSpec>& Neuropod::get_inputs() const
{
    return backend_->get_inputs();
}

const std::vector<TensorSpec>& Neuropod::get_outputs() const
{
    return backend_->get_outputs();
}

const std::string& Neuropod::get_name() const
{
   return model_config_->name;
}

const std::string& Neuropod::get_platform() const
{
   return model_config_->platform;
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
