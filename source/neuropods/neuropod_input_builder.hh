//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <memory>
#include <vector>

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/internal/tensor_store.hh"

namespace neuropods
{

// A NeuropodInputBuilder is used to supply input data to a Neuropod
class NeuropodInputBuilder final
{
private:
    // The backend used to allocate tensors
    std::shared_ptr<NeuropodBackend> backend_;

    // The tensor store that stores the created tensors
    std::unique_ptr<TensorStore> data_;

public:
    // You should not call this directly.
    // Use Neuropod::get_input_builder() instead
    explicit NeuropodInputBuilder(std::shared_ptr<NeuropodBackend> backend);
    ~NeuropodInputBuilder();

    // Add a tensor with a vector and shape
    // Returns a reference to the builder to allow easy chaining
    // Note: this method makes a copy of the vector
    template <typename T>
    NeuropodInputBuilder &add_tensor(const std::string &         node_name,
                                     const std::vector<T> &      input_data,
                                     const std::vector<int64_t> &input_dims);

    // Add a tensor with a pointer to existing data, a size, and a shape
    // Returns a reference to the builder to allow easy chaining
    // Note: this method makes a copy of the data
    template <typename T>
    NeuropodInputBuilder &add_tensor(const std::string &         node_name,
                                     const T *                   input_data,
                                     size_t                      input_data_size,
                                     const std::vector<int64_t> &input_dims);

    // Add a tensor of a certain shape
    // The InputBuilder will allocate the memory and return a
    // TypedNeuropodTensor<T> object that can be used to write the data.
    template <typename T>
    TypedNeuropodTensor<T> *allocate_tensor(const std::string &node_name, const std::vector<int64_t> &input_dims);

    // Get the data
    std::unique_ptr<TensorStore> build();
};

} // namespace neuropods
