//
// Uber, Inc. (c) 2018
//

#include "neuropod_input_builder.hh"

#include <cstring>

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/internal/tensor_store.hh"

namespace neuropods
{

struct NeuropodInputBuilder::impl
{
    // The backend used to allocate memory
    std::shared_ptr<NeuropodBackend> backend;

    // The tensor store that stores the created tensors
    std::unique_ptr<TensorStore> data;
};

NeuropodInputBuilder::NeuropodInputBuilder(std::shared_ptr<NeuropodBackend> backend) : pimpl(stdx::make_unique<impl>())
{
    pimpl->backend = std::move(backend);
    pimpl->data    = stdx::make_unique<TensorStore>();
}

NeuropodInputBuilder::~NeuropodInputBuilder() = default;


template <typename T>
NeuropodInputBuilder &NeuropodInputBuilder::add_tensor(const std::string &         node_name,
                                                       const std::vector<T> &      input_data,
                                                       const std::vector<int64_t> &input_dims)
{
    // Create a tensor with the underlying data
    return add_tensor(node_name, input_data.data(), input_data.size(), input_dims);
}

// Specialization for strings
template <>
NeuropodInputBuilder &NeuropodInputBuilder::add_tensor(const std::string &             node_name,
                                                       const std::vector<std::string> &input_data,
                                                       const std::vector<int64_t> &    input_dims)
{
    std::shared_ptr<NeuropodTensor> tensor = pimpl->backend->allocate_tensor(node_name, input_dims, STRING_TENSOR);

    // Add it to the vector of tensors stored in the builder
    pimpl->data->tensors.emplace_back(tensor);

    // Downcast to a TypedNeuropodTensor so we can set the data
    auto typed_tensor = tensor->as_typed_tensor<std::string>();

    // Set the data
    typed_tensor->set(input_data);

    // (for easy chaining)
    return *this;
}

template <typename T>
NeuropodInputBuilder &NeuropodInputBuilder::add_tensor(const std::string &         node_name,
                                                       const T *                   input_data,
                                                       size_t                      input_data_size,
                                                       const std::vector<int64_t> &input_dims)
{
    // Allocate a new tensor of the correct type
    auto *typed_tensor = allocate_tensor<T>(node_name, input_dims);
    T *   tensor_data  = static_cast<T *>(typed_tensor->get_raw_data_ptr());

    // Copy the data into the newly allocated memory
    std::memcpy(tensor_data, input_data, input_data_size * sizeof(T));

    // (for easy chaining)
    return *this;
}

template <typename T>
TypedNeuropodTensor<T> *NeuropodInputBuilder::allocate_tensor(const std::string &         node_name,
                                                              const std::vector<int64_t> &input_dims)
{
    if (pimpl->data->find(node_name))
    {
        std::stringstream error_message;
        error_message << "Tensor " << node_name << " was already created.";
        throw std::runtime_error(error_message.str());
    }

    std::shared_ptr<NeuropodTensor> tensor
        = pimpl->backend->allocate_tensor(node_name, input_dims, get_tensor_type_from_cpp<T>());

    // Add it to the vector of tensors stored in the builder
    pimpl->data->tensors.emplace_back(tensor);

    // Downcast to a TypedNeuropodTensor so we can get the data pointer
    return tensor->as_typed_tensor<T>();
}

std::unique_ptr<TensorStore> NeuropodInputBuilder::build()
{
    return std::move(pimpl->data);
}

// Instantiate the templates
#define INIT_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE)                                                                        \
    template NeuropodInputBuilder &NeuropodInputBuilder::add_tensor<CPP_TYPE>(const std::string &          node_name,           \
                                                                              const std::vector<CPP_TYPE> &input_data,          \
                                                                              const std::vector<int64_t> & input_dims);          \
                                                                                                                                \
    template NeuropodInputBuilder &         NeuropodInputBuilder::add_tensor<CPP_TYPE>(const std::string &node_name,            \
                                                                              const CPP_TYPE *   input_data,           \
                                                                              size_t             input_data_size,      \
                                                                              const std::vector<int64_t> &input_dims); \
    template TypedNeuropodTensor<CPP_TYPE> *NeuropodInputBuilder::allocate_tensor(                                              \
        const std::string &node_name, const std::vector<int64_t> &input_dims);

FOR_EACH_TYPE_MAPPING_EXCEPT_STRING(INIT_TEMPLATES_FOR_TYPE);

} // namespace neuropods
