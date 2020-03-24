//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropod/internal/error_utils_header_only.hh"
#include "neuropod/internal/memory_utils.hh"
#include "neuropod/internal/tensor_accessor.hh"
#include "neuropod/internal/type_macros.hh"
#include "neuropod/options.hh"

#include <cstring>
#include <functional>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuropod
{

namespace
{

// Utility to get a neuropod tensor type from a c++ type
template <typename T>
inline TensorType get_tensor_type_from_cpp() = delete;

#define GET_TENSOR_TYPE_FN(CPP_TYPE, NEUROPOD_TYPE)        \
    template <>                                            \
    inline TensorType get_tensor_type_from_cpp<CPP_TYPE>() \
    {                                                      \
        return NEUROPOD_TYPE;                              \
    }

FOR_EACH_TYPE_MAPPING_INCLUDING_STRING(GET_TENSOR_TYPE_FN)

// Utility to set the serialization tag for a class
// Note that the MAKE_SERIALIZABLE macro must be used
// in order to actually make a class serializable.
#define SET_SERIALIZE_TAG(tag)                                    \
    /* Used by the MAKE_SERIALIZABLE macro */                     \
    static std::string get_static_serialize_tag() { return tag; } \
    std::string        get_serialize_tag() const { return tag; }

} // namespace

namespace detail
{

// Overloads for error signatures we need in this file
[[noreturn]] void throw_error_hh(
    const char *file, int line, const char *function, const std::string &message, TensorType type);
[[noreturn]] void throw_error_hh(
    const char *file, int line, const char *function, const std::string &message, TensorType type1, TensorType type2);

} // namespace detail

// Forward declare NeuropodTensor and SealedNeuropodTensor
class NeuropodTensor;
class SealedNeuropodTensor;

// Forward declare TypedNeuropodTensor
template <typename T>
class TypedNeuropodTensor;

namespace internal
{

// A struct used internally to work with NeuropodTensors
struct NeuropodTensorRawDataAccess;

} // namespace internal

// Base value type for Neuropod
class NeuropodValue
{
private:
    // Whether or not this item is a tensor
    const bool is_tensor_;

public:
    NeuropodValue(bool is_tensor) : is_tensor_(is_tensor) {}

    virtual ~NeuropodValue() {}

    // Type-checked downcast to a NeuropodTensor
    NeuropodTensor *      as_tensor();
    const NeuropodTensor *as_tensor() const;

    // Type-checked downcast to a TypedNeuropodTensor
    template <typename T>
    TypedNeuropodTensor<T> *as_typed_tensor();

    template <typename T>
    const TypedNeuropodTensor<T> *as_typed_tensor() const;

    // This checks equality of contents, not of addresses or underlying implementations
    // (e.g. comparing a TorchNeuropodTensor and a TensorflowNeuropodTensor with identical
    // shapes, types, and content would return true)
    bool operator==(const NeuropodValue &other) const;

    // Don't override this manually
    // Use the SET_SERIALIZE_TAG macro instead
    virtual std::string get_serialize_tag() const = 0;

protected:
    void assure_tensor() const
    {
        if (!is_tensor_)
        {
            NEUROPOD_ERROR_HH("This NeuropodValue is expected to be a NeuropodTensor.");
        }
    }
};

// A type erased version of a TypedNeuropodTensor. See the documentation for
// TypedNeuropodTensor for more details.
// Backends should not extend this class directly for their tensor implementations
class NeuropodTensor : public NeuropodValue
{
private:
    // The type of the tensor
    const TensorType tensor_type_;

    // The dimensions of the tensor
    const std::vector<int64_t> dims_;

    // The strides of the tensor
    const std::vector<int64_t> strides_;

    // The number of elements in the tensor
    const size_t num_elements_;

public:
    // Create a NeuropodTensor with a type and dims
    NeuropodTensor(TensorType tensor_type, const std::vector<int64_t> dims);

    NeuropodTensor(const NeuropodTensor &) = delete;

    virtual ~NeuropodTensor() {}

    friend std::ostream &operator<<(std::ostream &out, const NeuropodTensor &tensor)
    {
        out << "NeuropodTensor with type ";
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

    // Get the number of elements in the tensor
    size_t get_num_elements() const { return num_elements_; }

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
            NEUROPOD_ERROR_HH("Tensor is expected to have shape of {1} to be casted to a scalar.");
        }
        return this->template as_typed_tensor<T>()->template accessor<1>()[0];
    }

    template <typename T>
    const T &as_scalar() const
    {
        this->template assure_type<T>();
        // TODO(yevgeni): should change to assure_rank(0)
        this->assure_rank(1);
        if (this->get_dims()[0] != 1)
        {
            NEUROPOD_ERROR_HH("Tensor is expected to have shape of {1} to be casted to a scalar.");
        }
        return this->template as_typed_tensor<T>()->template accessor<1>()[0];
    }

    // This checks equality of contents, not of addresses or underlying implementations
    // (e.g. comparing a TorchNeuropodTensor and a TensorflowNeuropodTensor with identical
    // shapes, types, and content would return true)
    bool operator==(const NeuropodTensor &other) const;

    bool operator==(const NeuropodValue &other) const
    {
        // Defer to NeuropodValue equality
        return other == *this;
    }

    SET_SERIALIZE_TAG("neuropodtensor")

protected:
    template <typename T>
    void assure_type() const
    {
        TensorType requested = get_tensor_type_from_cpp<T>();
        TensorType actual    = get_tensor_type();

        if (requested != actual)
        {
            NEUROPOD_ERROR_HH(
                "Tried to downcast tensor of type {} to a TypedNeuropodTensor of type {}", actual, requested);
        }
    }
    void assure_rank(size_t expected_rank) const
    {
        const auto & dims = this->get_dims();
        const size_t rank = dims.size();
        if (rank != expected_rank)
        {
            NEUROPOD_ERROR_HH("Tensor is expected to have rank of {} while the actual rank is {}", expected_rank, rank);
        }
    }

    // Get the strides of the tensor
    const std::vector<int64_t> &get_strides() const { return strides_; }

    // This struct is used internally
    friend internal::NeuropodTensorRawDataAccess;

    // Get a raw void * to the underlying data
    virtual void *      get_untyped_data_ptr()       = 0;
    virtual const void *get_untyped_data_ptr() const = 0;

    virtual size_t get_bytes_per_element() const = 0;

    // Seal this tensor, move to the appropriate device, and return the sealed tensor
    friend class Sealer;
    virtual std::shared_ptr<SealedNeuropodTensor> seal(NeuropodDevice device) = 0;
};

// A TypedNeuropodTensor is a NeuropodTensor of a specific type.
// Backends should extend this class directly for their tensor implementations
template <typename T>
class TypedNeuropodTensor : public NeuropodTensor
{
public:
    TypedNeuropodTensor(const std::vector<int64_t> dims) : NeuropodTensor(get_tensor_type_from_cpp<T>(), dims) {}

    virtual ~TypedNeuropodTensor() {}

    T *      get_raw_data_ptr() { return static_cast<T *>(get_untyped_data_ptr()); }
    const T *get_raw_data_ptr() const { return static_cast<const T *>(get_untyped_data_ptr()); }

    template <size_t N>
    TensorAccessor<T *, N> accessor()
    {
        static_assert(N > 0, "`accessor()` is used for indexing a tensors, for scalars use `as_scalar()`");
        this->assure_rank(N);
        return TensorAccessor<T *, N>(get_raw_data_ptr(), get_dims().data(), get_strides().data());
    }

    template <size_t N>
    TensorAccessor<const T *, N> accessor() const
    {
        static_assert(N > 0, "`accessor()` is used for indexing tensors, for scalars use `as_scalar()`");
        this->assure_rank(N);
        return TensorAccessor<const T *, N>(get_raw_data_ptr(), get_dims().data(), get_strides().data());
    }

    T &as_scalar() { return NeuropodTensor::as_scalar<T>(); }

    const T &as_scalar() const { return NeuropodTensor::as_scalar<T>(); }

    std::vector<T> get_data_as_vector() const
    {
        std::vector<T> out;

        // Get the size and a pointer to the data
        size_t   size         = get_num_elements();
        const T *data_pointer = get_raw_data_ptr();

        // Copy into the vector
        out.insert(out.end(), &data_pointer[0], &data_pointer[size]);

        return out;
    }

    void copy_from(const T *input_data, size_t input_data_size)
    {
        // Get the number of elements and a pointer to the data
        size_t numel        = get_num_elements();
        T *    data_pointer = get_raw_data_ptr();

        if (numel != input_data_size)
        {
            NEUROPOD_ERROR_HH("The size of the provided data does not match the number"
                              "of elements in the tensor.");
        }

        // Copy the data into the tensor
        std::memcpy(data_pointer, input_data, input_data_size * sizeof(T));
    }

    void copy_from(const std::vector<T> &input_data) { copy_from(input_data.data(), input_data.size()); }

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

protected:
    size_t get_bytes_per_element() const { return sizeof(T); }
};

// A specialization for strings
template <>
class TypedNeuropodTensor<std::string> : public NeuropodTensor
{
public:
    TypedNeuropodTensor(const std::vector<int64_t> dims) : NeuropodTensor(STRING_TENSOR, dims) {}

    virtual ~TypedNeuropodTensor() {}

    // We can't get a raw pointer from a string tensor
    // virtual std::string *get_raw_data_ptr() = 0;

    // Set the data in the string tensor
    virtual void set(const std::vector<std::string> &data) = 0;

    // Get the data in the string tensor
    std::vector<std::string> get_data_as_vector() const
    {
        // Setup the output vector
        const auto               numel = get_num_elements();
        std::vector<std::string> out(numel);

        // Copy the data in
        for (int i = 0; i < numel; i++)
        {
            out[i] = (*this)[i];
        }

        return out;
    }

    template <size_t N>
    TensorAccessor<const TypedNeuropodTensor<std::string> &, N> accessor() const
    {
        static_assert(N > 0, "`accessor()` is used for indexing tensors, for scalars use `as_scalar()`");
        this->assure_rank(N);
        return TensorAccessor<const TypedNeuropodTensor<std::string> &, N>(
            *this, get_dims().data(), get_strides().data());
    }

protected:
    template <typename Container, size_t N>
    friend class TensorAccessor;
    friend class NeuropodTensor;

    // Get a particular element
    // TODO(vip): Can we use absl::string_view instead?
    virtual const std::string operator[](size_t index) const = 0;

    // We can't get a raw pointer from a string tensor
    void *get_untyped_data_ptr() { NEUROPOD_ERROR_HH("`get_untyped_data_ptr` is not supported for string tensors"); };

    const void *get_untyped_data_ptr() const
    {
        NEUROPOD_ERROR_HH("`get_untyped_data_ptr` is not supported for string tensors");
    };

    size_t get_bytes_per_element() const
    {
        NEUROPOD_ERROR_HH("`get_bytes_per_element` is not supported for string tensors");
    };
};

// An opaque NeuropodTensor.
// This is used to implement things like early GPU copy
class SealedNeuropodTensor : public NeuropodValue
{
public:
    SealedNeuropodTensor();
    virtual ~SealedNeuropodTensor();

    // TODO(vip): how do we serialize sealed tensors (e.g. if they're on GPU)?
    SET_SERIALIZE_TAG("sealedneuropodtensor")
};

// Utility to make a tensor of a specific type
#define MAKE_TENSOR(CPP_TYPE, NEUROPOD_TYPE)                                              \
    case NEUROPOD_TYPE: {                                                                 \
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
        NEUROPOD_ERROR_HH("Unsupported tensor type: {}", tensor_type);
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
        NEUROPOD_ERROR_HH("Unsupported tensor type: {}", tensor_type);
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

    virtual T get_native_data() const = 0;
};

// A map from a tensor name to a pointer to a NeuropodValue
// This is the input and output type of `infer`
using NeuropodValueMap = std::unordered_map<std::string, std::shared_ptr<NeuropodValue>>;

} // namespace neuropod
