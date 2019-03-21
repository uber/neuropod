//
// Uber, Inc. (c) 2018
//

#pragma once

#include <cstring>
#include <functional>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "neuropods/internal/memory_utils.hh"
#include "neuropods/internal/type_macros.hh"

namespace neuropods
{

namespace
{

// Utility to get a neuropod tensor type from a c++ type
template <typename T>
TensorType get_tensor_type_from_cpp() = delete;

#define GET_TENSOR_TYPE_FN(CPP_TYPE, NEUROPOD_TYPE) \
    template <>                                     \
    [[gnu::unused]]                                 \
    TensorType get_tensor_type_from_cpp<CPP_TYPE>() \
    {                                               \
        return NEUROPOD_TYPE;                       \
    }

FOR_EACH_TYPE_MAPPING_INCLUDING_STRING(GET_TENSOR_TYPE_FN)

} // namespace

// Forward declare TypedNeuropodTensor
template <typename T>
class TypedNeuropodTensor;

// A type erased version of a TypedNeuropodTensor. See the documentation for
// TypedNeuropodTensor for more details.
// Backends should not extend this class directly for their tensor implementations
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

    NeuropodTensor(const NeuropodTensor &) = delete;

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
        const auto dims = get_dims();
        return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
    }

    TensorType get_tensor_type() const { return tensor_type_; }

    // Downcast a NeuropodTensor to a TypedNeuropodTensor of a specific type
    // The requested type is checked to make sure it matches the actual type
    template <typename T>
    TypedNeuropodTensor<T> *as_typed_tensor()
    {
        this->template assure_type<T>();
        return dynamic_cast<TypedNeuropodTensor<T> *>(this);
    }

    template <typename T>
    const TypedNeuropodTensor<T> *as_typed_tensor() const
    {
        this->template assure_type<T>();
        return dynamic_cast<const TypedNeuropodTensor<T> *>(this);
    }

    // Returns a modifiable reference to the scalar represented by this object. Maybe we will make
    // scalars be represented by a rank-0 object. For now we check it is rank(1) with dims[0]==1.
    // An exception is thrown if the dimensions of the tensors are not {1}
    template <typename T>
    T &as_scalar()
    {
        this->template assure_type<T>();
        // TODO(yevgeni): should change to assure_rank(0)
        this->assure_rank(1);
        if (this->get_dims()[0] != 1)
        {
            throw std::runtime_error("Tensor is expected to have shape of {1} to be"
                                     "casted to a scalar.");
        }
        return (*this->template as_typed_tensor<T>())[0];
    }

    template <typename T>
    const T &as_scalar() const
    {
        this->template assure_type<T>();
        // TODO(yevgeni): should change to assure_rank(0)
        this->assure_rank(1);
        if (this->get_dims()[0] != 1)
        {
            throw std::runtime_error("Tensor is expected to have shape of {1} to be"
                                     "casted to a scalar.");
        }
        return (*this->template as_typed_tensor<T>())[0];
    }

protected:
    template <typename T>
    void assure_type() const
    {
        TensorType requested = get_tensor_type_from_cpp<T>();
        TensorType actual    = get_tensor_type();

        if (requested != actual)
        {
            std::stringstream ss;
            ss << "Tried to downcast tensor \"" << get_name() << "\" of type " << actual
               << " to a TypedNeuropodTensor of type " << requested;
            throw std::runtime_error(ss.str());
        }
    }
    void assure_rank(size_t expected_rank) const
    {
        const auto   dims = this->get_dims();
        const size_t rank = dims.size();
        if (rank != expected_rank)
        {
            std::stringstream error_message;

            error_message << "Tensor '" << this->get_name() << " is expected to have rank of " << expected_rank
                          << " while the actual rank is " << rank;
            throw std::runtime_error(error_message.str());
        }
    }
};

// A TypedNeuropodTensor is a NeuropodTensor of a specific type.
// Backends should extend this class directly for their tensor implementations
template <typename T>
class TypedNeuropodTensor : public NeuropodTensor
{
public:
    TypedNeuropodTensor(const std::string &name, const std::vector<int64_t> dims)
        : NeuropodTensor(name, get_tensor_type_from_cpp<T>(), dims)
    {
    }

    virtual ~TypedNeuropodTensor() {}

    virtual T *      get_raw_data_ptr()       = 0;
    virtual const T *get_raw_data_ptr() const = 0;

    virtual const T &operator[](uint32_t r) const { return (*this)(r); }
    virtual T &      operator[](uint32_t r) { return (*this)(r); }

    virtual const T &operator()(uint32_t r) const
    {
        this->assure_rank(1);
        return get_raw_data_ptr()[r];
    }

    virtual T &operator()(uint32_t r)
    {
        this->assure_rank(1);
        return get_raw_data_ptr()[r];
    }

    virtual const T &operator()(uint32_t r, uint32_t c) const
    {
        this->assure_rank(2);
        return get_raw_data_ptr()[r * this->get_dims()[1] + c];
    }

    virtual T &operator()(uint32_t r, uint32_t c)
    {
        this->assure_rank(2);
        return get_raw_data_ptr()[r * this->get_dims()[1] + c];
    }

    const T *begin() const
    {
        assure_rank(1);
        return &get_raw_data_ptr()[0];
    }

    T *begin()
    {
        assure_rank(1);
        return &get_raw_data_ptr()[0];
    }

    const T *end() const
    {
        assure_rank(1);
        return &get_raw_data_ptr()[get_dims()[0]];
    }

    T *end()
    {
        assure_rank(1);
        return &get_raw_data_ptr()[get_dims()[0]];
    }

    T &as_scalar() { return NeuropodTensor::as_scalar<T>(); }

    const T &as_scalar() const { return NeuropodTensor::as_scalar<T>(); }

    std::vector<T> get_data_as_vector()
    {
        std::vector<T> out;

        // Get the size and a pointer to the data
        size_t   size         = get_num_elements();
        const T *data_pointer = get_raw_data_ptr();

        // Copy into the vector
        out.insert(out.end(), &data_pointer[0], &data_pointer[size]);

        return out;
    }

    void copy_from(const T * input_data, size_t input_data_size)
    {
        // Get the number of elements and a pointer to the data
        size_t numel        = get_num_elements();
        T *    data_pointer = get_raw_data_ptr();

        if (numel != input_data_size)
        {
            throw std::runtime_error("The size of the provided data does not match the number"
                "of elements in the tensor.");
        }

        // Copy the data into the tensor
        std::memcpy(data_pointer, input_data, input_data_size * sizeof(T));
    }

    void copy_from(const std::vector<T> &input_data)
    {
        copy_from(input_data.data(), input_data.size());
    }

    friend std::ostream &operator<<(std::ostream &out, const TypedNeuropodTensor<T> &tensor)
    {
        out << static_cast<const NeuropodTensor &>(tensor) << std::endl;

        out << '[';

        const auto num_elements = tensor.get_num_elements();
        const T *  ptr          = tensor.get_raw_data_ptr();
        if (num_elements < 6)
        {
            for (size_t i = 0; i < num_elements; ++i)
            {
                if (i > 0)
                {
                    out << ", ";
                }
                out << +ptr[i];
            }
        }
        else
        {
            for (size_t i = 0; i < 3; ++i)
            {
                if (i > 0)
                {
                    out << ", ";
                }
                out << +ptr[i];
            }
            out << " ... ";
            for (size_t i = num_elements - 3; i < num_elements; ++i)
            {
                if (i > num_elements - 3)
                {
                    out << ", ";
                }
                out << +ptr[i];
            }
        }
        out << ']';
        return out;
    }
};

// A specialization for strings
template <>
class TypedNeuropodTensor<std::string> : public NeuropodTensor
{
public:
    TypedNeuropodTensor(const std::string &name, const std::vector<int64_t> dims)
        : NeuropodTensor(name, STRING_TENSOR, dims)
    {
    }

    virtual ~TypedNeuropodTensor() {}

    // We can't get a raw pointer from a string tensor
    // virtual std::string *get_raw_data_ptr() = 0;

    // TODO(vip): make this pure virtual once all the existing backends have
    // implementations.
    // Set the data in the string tensor
    virtual void set(const std::vector<std::string> &data)
    {
        throw std::runtime_error("Children must implement `set`");
    }

    // TODO(vip): make this pure virtual once all the existing backends have
    // implementations.
    // Get the data in the string tensor
    virtual std::vector<std::string> get_data_as_vector()
    {
        throw std::runtime_error("Children must implement `get_data_as_vector`");
    };
};


// Utility to make a tensor of a specific type
#define MAKE_TENSOR(CPP_TYPE, NEUROPOD_TYPE)                                              \
    case NEUROPOD_TYPE:                                                                   \
    {                                                                                     \
        return stdx::make_unique<TensorClass<CPP_TYPE>>(std::forward<Params>(params)...); \
    }

template <template <class> class TensorClass, typename... Params>
std::unique_ptr<NeuropodTensor> make_tensor(TensorType tensor_type, Params &&... params)
{
    // Make a tensor of the correct type and return it
    switch (tensor_type)
    {
        FOR_EACH_TYPE_MAPPING_INCLUDING_STRING(MAKE_TENSOR)
    default:
        throw std::runtime_error("Unsupported tensor type!");
    }
}

template <template <class> class TensorClass, typename... Params>
std::unique_ptr<NeuropodTensor> make_tensor_no_string(TensorType tensor_type, Params &&... params)
{
    // Make a tensor of the correct type and return it
    switch (tensor_type)
    {
        FOR_EACH_TYPE_MAPPING_EXCEPT_STRING(MAKE_TENSOR)
    default:
        throw std::runtime_error("Unsupported tensor type!");
    }
}

#undef MAKE_TENSOR

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
