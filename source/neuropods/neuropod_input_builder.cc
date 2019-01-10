//
// Uber, Inc. (c) 2018
//

#include "neuropod_input_builder.hh"

#include <cstring>

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/internal/neuropod_input_data.hh"
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

NeuropodInputBuilder::NeuropodInputBuilder(std::shared_ptr<NeuropodBackend> backend) : pimpl(std::make_unique<impl>())
{
    pimpl->backend = std::move(backend);
    pimpl->data    = std::make_unique<TensorStore>();
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


template <typename T>
NeuropodInputBuilder &NeuropodInputBuilder::add_tensor(const std::string &         node_name,
                                                       const T *                   input_data,
                                                       size_t                      input_data_size,
                                                       const std::vector<int64_t> &input_dims)
{
    // Allocate a new tensor of the correct type
    T *tensor_data = allocate_tensor<T>(node_name, input_data_size, input_dims);

    // Copy the data into the newly allocated memory
    std::memcpy(tensor_data, input_data, input_data_size * sizeof(T));

    // (for easy chaining)
    return *this;
}

template <typename T>
T *NeuropodInputBuilder::allocate_tensor(const std::string &         node_name,
                                         size_t                      input_data_size,
                                         const std::vector<int64_t> &input_dims)
{
    std::shared_ptr<NeuropodTensor> tensor
        = pimpl->backend->allocate_tensor(node_name, input_dims, get_tensor_type_from_cpp<T>());

    // Add it to the vector of tensors stored in the builder
    pimpl->data->tensors.emplace_back(tensor);

    // Downcast to a TypedNeuropodTensor so we can get the data pointer
    auto typed_tensor = tensor->as_typed_tensor<T>();

    return typed_tensor->get_raw_data_ptr();
}

std::unique_ptr<NeuropodInputData, NeuropodInputDataDeleter> NeuropodInputBuilder::build()
{
    // Can't use make_unique because of a custom deleter
    std::unique_ptr<NeuropodInputData, NeuropodInputDataDeleter> out(new NeuropodInputData());
    out->data = std::move(pimpl->data);
    return out;
}

// Instantiate the templates
#define INIT_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE)                                                               \
    template NeuropodInputBuilder &NeuropodInputBuilder::add_tensor<CPP_TYPE>(const std::string &          node_name,  \
                                                                              const std::vector<CPP_TYPE> &input_data, \
                                                                              const std::vector<int64_t> & input_dims); \
                                                                                                                       \
    template NeuropodInputBuilder &NeuropodInputBuilder::add_tensor<CPP_TYPE>(const std::string &node_name,            \
                                                                              const CPP_TYPE *   input_data,           \
                                                                              size_t             input_data_size,      \
                                                                              const std::vector<int64_t> &input_dims); \
                                                                                                                       \
    template CPP_TYPE *NeuropodInputBuilder::allocate_tensor(                                                          \
        const std::string &node_name, size_t input_data_size, const std::vector<int64_t> &input_dims);

FOR_EACH_TYPE_MAPPING(INIT_TEMPLATES_FOR_TYPE);

} // namespace neuropods
