//
// Uber, Inc. (c) 2018
//

#pragma once

#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "type_macros.hh"

// Used below to create a variant that supports all the neuropod-supported pointer types
#define PTR(CPP_TYPE, NEUROPOD_TYPE) CPP_TYPE *

namespace neuropods
{

namespace
{

// Utility to get a neuropod tensor type from a c++ type
template <typename T>
TensorType get_tensor_type_from_cpp()
{
}

#define GET_TENSOR_TYPE_FN(CPP_TYPE, NEUROPOD_TYPE) \
    template <>                                     \
    TensorType get_tensor_type_from_cpp<CPP_TYPE>() \
    {                                               \
        return NEUROPOD_TYPE;                       \
    }

FOR_EACH_TYPE_MAPPING(GET_TENSOR_TYPE_FN)

} // namespace

// Forward declare TypedNeuropodTensor
template <typename T>
class TypedNeuropodTensor;

// A type erased version of a TypedNeuropodTensor. See the documentation for
// TypedNeuropodTensor for more details.
// Backends should not extend this class directly for their tensor implementations
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

    // Downcast a NeuropodTensor to a TypedNeuropodTensor of a specific type
    // The requested type is checked to make sure it matches the actual type
    template <typename T>
    TypedNeuropodTensor<T> *as_typed_tensor()
    {
        TensorType requested = get_tensor_type_from_cpp<T>();
        TensorType actual    = get_tensor_type();

        if (requested != actual)
        {
            std::stringstream ss;
            ss << "Tried to downcast a tensor of type ";
            ss << actual;
            ss << " to a TypedNeuropodTensor of type ";
            ss << requested;
            throw std::runtime_error(ss.str());
        }

        return static_cast<TypedNeuropodTensor<T> *>(this);
    }
};

// A TypedNeuropodTensor is a NeuropodTensor of a specific type.
// Backends should extend this class directly for their tensor implementations
// Note: This class is internal to neuropods and should not be exposed to users
template <typename T>
class TypedNeuropodTensor : public NeuropodTensor
{
public:
    TypedNeuropodTensor(const std::string &name, const std::vector<int64_t> dims)
        : NeuropodTensor(name, get_tensor_type_from_cpp<T>(), dims)
    {
    }

    virtual ~TypedNeuropodTensor() {}

    virtual T *get_raw_data_ptr() = 0;
};

// Utility to make a tensor of a specific type
template <template <class> class TensorClass, typename... Params>
std::unique_ptr<NeuropodTensor> make_tensor(TensorType tensor_type, Params &&... params)
{
#define MAKE_TENSOR(CPP_TYPE, NEUROPOD_TYPE)                                              \
    case NEUROPOD_TYPE:                                                                   \
    {                                                                                     \
        return std::make_unique<TensorClass<CPP_TYPE>>(std::forward<Params>(params)...); \
    }

    // Make a tensor of the correct type and return it
    switch (tensor_type)
    {
        FOR_EACH_TYPE_MAPPING(MAKE_TENSOR)
    }
#undef MAKE_TENSOR
}

// Utility superclass for getting data from a tensor
// without having to downcast to a specific TypedNeuropodTensor
template <typename T>
class NativeDataContainer
{
public:
    NativeDataContainer() {}
    virtual ~NativeDataContainer() {}

    virtual T get_native_data() = 0;
};

} // namespace neuropods
