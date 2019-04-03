//
// Uber, Inc. (c) 2018
//

#include "neuropods.hh"

#include "neuropods/internal/backend_registration.hh"
#include "neuropods/internal/config_utils.hh"
#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/internal/tensor_store.hh"

namespace neuropods
{

Neuropod::Neuropod(const std::string &neuropod_path)
    : Neuropod(neuropod_path, std::unordered_map<std::string, std::string>())
{
}

// Find the right backend to use and load the neuropod
Neuropod::Neuropod(const std::string &neuropod_path, const std::unordered_map<std::string, std::string> &default_backend_overrides)
{
    // Load the model config and find the correct backend
    auto model_config = load_model_config(neuropod_path);
    auto platform = model_config->platform;

    backend_ = get_backend_for_type(default_backend_overrides, platform)(neuropod_path, std::move(model_config));
}

// Load the neuropod using the specified backend
Neuropod::Neuropod(const std::string &neuropod_path, const std::string &backend_name)
    : backend_(get_backend_by_name(backend_name)(neuropod_path, load_model_config(neuropod_path)))
{
}

// Use the backend that was provided by the user
Neuropod::Neuropod(std::shared_ptr<NeuropodBackend> backend)
    : backend_(backend)
{
}

Neuropod::~Neuropod() = default;

std::unique_ptr<TensorStore> Neuropod::infer(const std::unordered_set<std::shared_ptr<NeuropodTensor>> &inputs)
{
    // Run inference
    // TODO(vip): make sure that tensor names in `inputs` are not repeated
    // TODO(vip): if `has_input_and_output_spec()`, do shape/type checking before calling `infer`
    return backend_->infer(inputs);
}

bool Neuropod::has_input_and_output_spec() const
{
    return backend_->get_neuropod_model_config() != nullptr;
}

const std::vector<TensorSpec> &Neuropod::get_inputs() const
{
    if (!has_input_and_output_spec())
    {
        NEUROPOD_ERROR("This neuropod does not have input and output specs. "
            "Please check `has_input_and_output_spec()` before calling `get_outputs()`");
    }

    return backend_->get_neuropod_model_config()->inputs;
}

const std::vector<TensorSpec> &Neuropod::get_outputs() const
{
    if (!has_input_and_output_spec())
    {
        NEUROPOD_ERROR("This neuropod does not have input and output specs. "
            "Please check `has_input_and_output_spec()` before calling `get_outputs()`");
    }

    return backend_->get_neuropod_model_config()->outputs;
}

template <typename T>
std::shared_ptr<TypedNeuropodTensor<T>> Neuropod::allocate_tensor(
    const std::string &node_name,
    const std::vector<int64_t> &input_dims)
{
    std::shared_ptr<NeuropodTensor> tensor
        = backend_->allocate_tensor(node_name, input_dims, get_tensor_type_from_cpp<T>());

    return std::dynamic_pointer_cast<TypedNeuropodTensor<T>>(tensor);
}

template <typename T>
std::shared_ptr<TypedNeuropodTensor<T>> Neuropod::tensor_from_memory(
    const std::string &         node_name,
    const std::vector<int64_t> &input_dims,
    T *                         data,
    const Deleter &             deleter)
{
    std::shared_ptr<NeuropodTensor> tensor
        = backend_->tensor_from_memory(node_name, input_dims, get_tensor_type_from_cpp<T>(), data, deleter);

    return std::dynamic_pointer_cast<TypedNeuropodTensor<T>>(tensor);
}

// Instantiate the templates
#define INIT_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE)                                                                          \
    template std::shared_ptr<TypedNeuropodTensor<CPP_TYPE>> Neuropod::tensor_from_memory(const std::string &         node_name,   \
                                                                                         const std::vector<int64_t> &input_dims,  \
                                                                                         CPP_TYPE *                  data,        \
                                                                                         const Deleter &             deleter);


#define INIT_STRING_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE)                             \
    template std::shared_ptr<TypedNeuropodTensor<CPP_TYPE>> Neuropod::allocate_tensor(      \
        const std::string &node_name, const std::vector<int64_t> &input_dims);

FOR_EACH_TYPE_MAPPING_EXCEPT_STRING(INIT_TEMPLATES_FOR_TYPE);
FOR_EACH_TYPE_MAPPING_INCLUDING_STRING(INIT_STRING_TEMPLATES_FOR_TYPE);

} // namespace neuropods
