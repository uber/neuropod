//
// Uber, Inc. (c) 2018
//

#include "neuropod_output_data.hh"

#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/internal/tensor_store.hh"

namespace neuropods
{

NeuropodOutputData::NeuropodOutputData(std::unique_ptr<TensorStore> tensor_store)
    : tensor_store(std::move(tensor_store))
{
}

NeuropodOutputData::~NeuropodOutputData() = default;

template <typename T>
void NeuropodOutputData::get_data_pointer_and_size(const std::string &node_name, const T *&pointer, size_t &size) const
{
    auto tensor = tensor_store->find(node_name);

    // Downcast to a TypedNeuropodTensor so we can get the data pointer
    auto typed_tensor = tensor->as_typed_tensor<T>();

    pointer = typed_tensor->get_raw_data_ptr();
    size    = tensor->get_num_elements();
}

template <typename T>
std::vector<T> NeuropodOutputData::get_data_as_vector(const std::string &node_name) const
{
    std::vector<T> out;

    // Get the size and a pointer to the data
    size_t   size;
    const T *data_pointer;
    get_data_pointer_and_size(node_name, data_pointer, size);

    // Copy into the vector
    out.insert(out.end(), &data_pointer[0], &data_pointer[size]);

    return out;
}

std::vector<int64_t> NeuropodOutputData::get_shape(const std::string &node_name) const
{
    return tensor_store->find(node_name)->get_dims();
}

// Instantiate the templates
#define INIT_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE)                             \
    template void NeuropodOutputData::get_data_pointer_and_size(                     \
        const std::string &node_name, const CPP_TYPE *&pointer, size_t &size) const; \
    template std::vector<CPP_TYPE> NeuropodOutputData::get_data_as_vector(const std::string &node_name) const;

FOR_EACH_TYPE_MAPPING_EXCEPT_STRING(INIT_TEMPLATES_FOR_TYPE);

} // namespace neuropods
