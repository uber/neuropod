//
// Uber, Inc. (c) 2018
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "neuropods/fwd_declarations.hh"

namespace neuropods
{

// The return type of `infer`
class NeuropodOutputData
{
protected:
    // Hidden internal storage
    std::unique_ptr<TensorStore> tensor_store;

public:
    explicit NeuropodOutputData(std::unique_ptr<TensorStore> tensor_store);
    ~NeuropodOutputData();

    // Get a pointer to the underlying data
    template <typename T>
    void get_data_pointer_and_size(const std::string &node_name, const T *&pointer, size_t &size) const;

    // Get the data as a vector
    // Note: this operation requires a copy
    template <typename T>
    std::vector<T> get_data_as_vector(const std::string &node_name) const;

    // Get the shape of a tensor
    std::vector<int64_t> get_shape(const std::string &node_name) const;
};

} // namespace neuropods
