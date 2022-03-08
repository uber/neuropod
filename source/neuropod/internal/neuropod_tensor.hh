/* Copyright (c) 2020 The Neuropod Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
    std::string        get_serialize_tag() const override { return tag; }

} // namespace

namespace detail
{

// Overloads for error signatures we need in this file
[[noreturn]] void throw_error_hh(
    const char *file, int line, const char *function, const std::string &message, TensorType type);
[[noreturn]] void throw_error_hh(
    const char *file, int line, const char *function, const std::string &message, TensorType type1, TensorType type2);

// Utility to compute strides given a vector of dimensions
std::vector<int64_t> compute_strides(const std::vector<int64_t> &dims);

} // namespace detail

// Forward declare NeuropodTensor
class NeuropodTensor;

// Forward declare TypedNeuropodTensor
template <typename T>
class TypedNeuropodTensor;

namespace internal
{

// A struct used internally to work with NeuropodTensors
struct NeuropodTensorRawDataAccess;

} // namespace internal

// Base value type for Neuropod
class NeuropodValue : public std::enable_shared_from_this<NeuropodValue>
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

    // The device of this tensor
    const NeuropodDevice device_;

public:
    // Create a NeuropodTensor with a type and dims
    NeuropodTensor(TensorType tensor_type, const std::vector<int64_t> dims, NeuropodDevice device = Device::CPU);

    NeuropodTensor(const NeuropodTensor &) = delete;

    virtual ~NeuropodTensor() override {}

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
        out << " on device " << tensor.device_;
        return out;
    }

    // Get the dimensions of the tensor
    const std::vector<int64_t> &get_dims() const { return dims_; }

    // Get the number of elements in the tensor
    size_t get_num_elements() const { return num_elements_; }

    TensorType get_tensor_type() const { return tensor_type_; }

    NeuropodDevice get_device() const { return device_; }

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

    void assure_device_cpu() const;

    // Check that requested_dims is compatible with the current tensor
    void assure_view_compatible_shape(const std::vector<int64_t> &requested_dims) const;

    template <typename Container, typename... Dim>
    auto view_helper(Container data, Dim... requested_dims) const
    {
        this->assure_device_cpu();
        std::vector<int64_t> new_dims = {requested_dims...};

        // Check that the requested dims are compatible with this tensor
        assure_view_compatible_shape(new_dims);

        // Create and return a tensor view
        constexpr size_t num_dims    = sizeof...(requested_dims);
        auto             new_strides = detail::compute_strides(new_dims);
        return TensorView<Container, num_dims>(data, std::move(new_dims), std::move(new_strides));
    }

    // Seal a tensor
    // This is used to implement things like early GPU copy
    friend class Sealer;

    // Copy a tensor to a particular device and return it
    // If this tensor is already on the target device, this is a noop
    std::shared_ptr<NeuropodValue> to(NeuropodDevice device)
    {
        if (device == device_)
        {
            // Already on the target device
            return this->shared_from_this();
        }

        return to_internal(device);
    }

    // TODO(vip): make this pure virtual once we have implementations for all backends
    virtual std::shared_ptr<NeuropodValue> to_internal(NeuropodDevice /*unused*/) { return this->shared_from_this(); }

    // Get the strides of the tensor
    const std::vector<int64_t> &get_strides() const { return strides_; }

    // This struct is used internally
    friend internal::NeuropodTensorRawDataAccess;

    // Get a raw void * to the underlying data
    virtual void *      get_untyped_data_ptr()       = 0;
    virtual const void *get_untyped_data_ptr() const = 0;

    virtual size_t get_bytes_per_element() const = 0;
};

// A TypedNeuropodTensor is a NeuropodTensor of a specific type.
// Backends should extend this class directly for their tensor implementations
template <typename T>
class TypedNeuropodTensor : public NeuropodTensor
{
public:
    TypedNeuropodTensor(const std::vector<int64_t> dims) : NeuropodTensor(get_tensor_type_from_cpp<T>(), dims) {}

    virtual ~TypedNeuropodTensor() {}

    T *get_raw_data_ptr()
    {
        this->assure_device_cpu();
        return static_cast<T *>(get_untyped_data_ptr());
    }

    const T *get_raw_data_ptr() const
    {
        this->assure_device_cpu();
        return static_cast<const T *>(get_untyped_data_ptr());
    }

    template <size_t N>
    TensorAccessor<T *, N> accessor()
    {
        static_assert(N > 0, "`accessor()` is used for indexing a tensors, for scalars use `as_scalar()`");
        this->assure_device_cpu();
        this->assure_rank(N);
        return TensorAccessor<T *, N>(get_raw_data_ptr(), get_dims().data(), get_strides().data());
    }

    template <size_t N>
    TensorAccessor<const T *, N> accessor() const
    {
        static_assert(N > 0, "`accessor()` is used for indexing tensors, for scalars use `as_scalar()`");
        this->assure_device_cpu();
        this->assure_rank(N);
        return TensorAccessor<const T *, N>(get_raw_data_ptr(), get_dims().data(), get_strides().data());
    }

    // Return a view of this tensor with the requested dimensions.
    // `requested_dims` must be compatible with the dims of this tensor.
    // This method does not make a copy of the tensor and changes made in the
    // returned view are visible in the original tensor.
    template <typename... Dim>
    auto view(Dim... requested_dims)
    {
        return view_helper(get_raw_data_ptr(), std::forward<Dim>(requested_dims)...);
    }

    template <typename... Dim>
    auto view(Dim... requested_dims) const
    {
        return view_helper(get_raw_data_ptr(), std::forward<Dim>(requested_dims)...);
    }

    // Return a view of this tensor as a flat tensor
    // (a 1D tensor)
    auto flat()
    {
        // Return a view with 1 dim
        return view(static_cast<int64_t>(get_num_elements()));
    }

    auto flat() const
    {
        // Return a view with 1 dim
        return view(static_cast<int64_t>(get_num_elements()));
    }

    T &as_scalar()
    {
        this->assure_device_cpu();
        return NeuropodTensor::as_scalar<T>();
    }

    const T &as_scalar() const
    {
        this->assure_device_cpu();
        return NeuropodTensor::as_scalar<T>();
    }

    std::vector<T> get_data_as_vector() const
    {
        this->assure_device_cpu();
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
        this->assure_device_cpu();
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

        if (tensor.get_device() != Device::CPU)
        {
            return out;
        }

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

// A class that lets us implement read/write accessors for strings
// It can be implicitly converted to std::string and std::strings can be assigned to it
// It proxies assignments/conversions to T::set and T::get respectively.
// This is necessary because there isn't a standard string tensor format used by all frameworks
// Accessors for string tensors return a StringProxy instead of an std::string directly
template <typename T>
class StringProxy
{
private:
    T &    tensor_;
    size_t index_;

public:
    StringProxy(T &tensor, size_t index) : tensor_(tensor), index_(index) {}

    void operator=(const std::string &value) { tensor_.set(index_, value); }

    // Allow implicit conversions to string
    operator std::string() const { return tensor_.get(index_); }

    // Foward all the below operators to std::string::compare
    static int cmp(const std::string &lhs, const std::string &rhs) { return lhs.compare(rhs); }

    // Based on https://en.cppreference.com/w/cpp/language/operators
    // TODO(vip): Simplify these operators
    friend bool operator==(const StringProxy &lhs, const std::string &rhs) { return cmp(lhs, rhs) == 0; }
    friend bool operator==(const std::string &lhs, const StringProxy &rhs) { return cmp(lhs, rhs) == 0; }
    friend bool operator!=(const StringProxy &lhs, const std::string &rhs) { return cmp(lhs, rhs) != 0; }
    friend bool operator<(const StringProxy &lhs, const std::string &rhs) { return cmp(lhs, rhs) < 0; }
    friend bool operator>(const StringProxy &lhs, const std::string &rhs) { return cmp(lhs, rhs) > 0; }
    friend bool operator<=(const StringProxy &lhs, const std::string &rhs) { return cmp(lhs, rhs) <= 0; }
    friend bool operator>=(const StringProxy &lhs, const std::string &rhs) { return cmp(lhs, rhs) >= 0; }
};

// A specialization for strings
template <>
class TypedNeuropodTensor<std::string> : public NeuropodTensor
{
public:
    TypedNeuropodTensor(const std::vector<int64_t> dims) : NeuropodTensor(STRING_TENSOR, dims) {}

    virtual ~TypedNeuropodTensor() {}

    // We can't get a raw pointer from a string tensor
    // TODO(vip): this->assure_device_cpu();
    // virtual std::string *get_raw_data_ptr() = 0;

    // Set the data in the string tensor
    virtual void copy_from(const std::vector<std::string> &data) = 0;

    // Get the data in the string tensor
    std::vector<std::string> get_data_as_vector() const
    {
        this->assure_device_cpu();

        // Setup the output vector
        const auto               numel = get_num_elements();
        std::vector<std::string> out(numel);

        // Copy the data in
        for (size_t i = 0; i < numel; i++)
        {
            out[i] = get(i);
        }

        return out;
    }

    template <size_t N>
    TensorAccessor<TypedNeuropodTensor<std::string> &, N> accessor()
    {
        static_assert(N > 0, "`accessor()` is used for indexing tensors, for scalars use `as_scalar()`");
        this->assure_device_cpu();
        this->assure_rank(N);
        return TensorAccessor<TypedNeuropodTensor<std::string> &, N>(*this, get_dims().data(), get_strides().data());
    }

    template <size_t N>
    TensorAccessor<const TypedNeuropodTensor<std::string> &, N> accessor() const
    {
        static_assert(N > 0, "`accessor()` is used for indexing tensors, for scalars use `as_scalar()`");
        this->assure_device_cpu();
        this->assure_rank(N);
        return TensorAccessor<const TypedNeuropodTensor<std::string> &, N>(
            *this, get_dims().data(), get_strides().data());
    }

    // Return a view of this tensor with the requested dimensions.
    // `requested_dims` must be compatible with the dims of this tensor.
    // This method does not make a copy of the tensor and changes made in the
    // returned view are visible in the original tensor.
    template <typename... Dim>
    auto view(Dim... requested_dims)
    {
        return view_helper<TypedNeuropodTensor<std::string> &>(*this, std::forward<Dim>(requested_dims)...);
    }

    template <typename... Dim>
    auto view(Dim... requested_dims) const
    {
        return view_helper<const TypedNeuropodTensor<std::string> &>(*this, std::forward<Dim>(requested_dims)...);
    }

    // Return a view of this tensor as a flat tensor
    // (a 1D tensor)
    auto flat()
    {
        // Return a view with 1 dim
        return view(static_cast<int64_t>(get_num_elements()));
    }

    auto flat() const
    {
        // Return a view with 1 dim
        return view(static_cast<int64_t>(get_num_elements()));
    }

protected:
    template <typename Container, size_t N>
    friend class TensorAccessor;
    friend class NeuropodTensor;

    // Get a particular element
    template <typename T>
    friend class StringProxy;
    const StringProxy<const TypedNeuropodTensor<std::string>> operator[](size_t index) const
    {
        return StringProxy<const TypedNeuropodTensor<std::string>>(*this, index);
    }

    StringProxy<TypedNeuropodTensor<std::string>> operator[](size_t index)
    {
        return StringProxy<TypedNeuropodTensor<std::string>>(*this, index);
    }

    // This might be slow (virtual calls in tight loops + maybe backends redoing work)
    // TODO(vip): Profile and maybe add a write-through cache
    virtual std::string get(size_t index) const                     = 0;
    virtual void        set(size_t index, const std::string &value) = 0;

    // We can't get a raw pointer from a string tensor
    void *get_untyped_data_ptr() { NEUROPOD_ERROR_HH("`get_untyped_data_ptr` is not supported for string tensors"); }

    const void *get_untyped_data_ptr() const
    {
        NEUROPOD_ERROR_HH("`get_untyped_data_ptr` is not supported for string tensors");
    }

    size_t get_bytes_per_element() const
    {
        NEUROPOD_ERROR_HH("`get_bytes_per_element` is not supported for string tensors");
    }
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
    }
}

template <template <class> class TensorClass, typename... Params>
std::unique_ptr<NeuropodTensor> make_tensor_no_string(TensorType tensor_type, Params &&... params)
{
    // Make a tensor of the correct type and return it
    switch (tensor_type)
    {
        FOR_EACH_TYPE_MAPPING_EXCEPT_STRING(MAKE_TENSOR)
    case STRING_TENSOR:
        NEUROPOD_ERROR_HH("`make_tensor_no_string` does not support type STRING_TENSOR");
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

// A map from a tensor name to a pointer to a NeuropodValue
// This is the input and output type of `infer`
using NeuropodValueMap = std::unordered_map<std::string, std::shared_ptr<NeuropodValue>>;

} // namespace neuropod
