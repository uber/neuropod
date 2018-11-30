//
// Uber, Inc. (c) 2018
//

#pragma once

#include <boost/variant.hpp>
#include <string>
#include <vector>

#include "type_macros.hh"

// Used below to create a variant that supports all the neuropod-supported pointer types
#define PTR(CPP_TYPE, NEUROPOD_TYPE) CPP_TYPE *

namespace neuropods
{

// All the supported types of data for tensors
typedef boost::variant<FOR_EACH_TYPE_MAPPING_DELIM(PTR, COMMA_DELIM)> TensorDataPointer;

// Each backend implements a subclass of this class
// Note: This class is internal to neuropods and should not be exposed to users
class NeuropodTensor
{
private:
    // The name of the tensor
    const std::string name_;

    // The type of the tensor
    const TensorType tensor_type_;

    // The dimensions of the tensor
    const std::vector<int64_t> dims_;

public:
    // Create a NeuropodTensor with a name and type
    NeuropodTensor(const std::string &name, TensorType tensor_type, const std::vector<int64_t> dims)
        : name_(name), tensor_type_(tensor_type), dims_(dims)
    {
    }

    virtual ~NeuropodTensor() {}

    friend std::ostream &operator<<(std::ostream &out, const NeuropodTensor &tensor)
    {
        out << "NeuropodTensor '" << tensor.get_name() << "' with type ";
        out << tensor.get_tensor_type();
        out << " and shape (";
        for (const int64_t dim : tensor.get_dims())
        {
            out << dim << ", ";
        }

        out << ")";
        return out;
    }

    // Get a pointer to the underlying data
    virtual TensorDataPointer get_data_ptr() = 0;

    // Get the dimensions of the tensor
    const std::vector<int64_t> &get_dims() const { return dims_; }

    // Get the name of the tensor
    const std::string &get_name() const { return name_; }

    // Get the number of elements in the tensor
    // by multiplying all the dims together
    size_t get_num_elements() const
    {
        size_t tensor_size = 1;
        for (const auto dim_size : get_dims())
        {
            tensor_size *= dim_size;
        }

        return tensor_size;
    }

    TensorType get_tensor_type() const { return tensor_type_; }
};

} // namespace neuropods
