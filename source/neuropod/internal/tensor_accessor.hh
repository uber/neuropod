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

#include <vector>

namespace neuropod
{

// An iterator type used by the accessors
// This is a simple utility class that acts as an iterator for any subscriptable class
template <typename Accessor>
class AccessorIterator
{
private:
    Accessor *accessor_;
    int64_t   index_;

public:
    using value_type = decltype((*accessor_)[index_]);

    AccessorIterator(Accessor *accessor, int64_t index = 0) : accessor_(accessor), index_(index) {}

    AccessorIterator<Accessor> &operator++()
    {
        index_++;
        return *this;
    }

    // * returns the same type as indexing using the accessor directly
    // We're using `auto` and `decltype` here so we can support multiple accessor types
    // in a generic way.
    auto operator*() const -> decltype((*accessor_)[index_]) { return (*accessor_)[index_]; }

    bool operator!=(const AccessorIterator<Accessor> &other) const
    {
        return index_ != other.index_ || accessor_ != other.accessor_;
    }
};

// Utility function to create an iterator
template <typename T>
inline AccessorIterator<T> get_iterator(T *accessor, size_t index = 0)
{
    return AccessorIterator<T>(accessor, index);
}

// `TensorAccessor`s are used to access data in a NeuropodTensor. They are very efficient and are comparable
// to raw pointer operations during an optimized build (see `benchmark_accessor.cc`). They can be used as follows:
//
//     auto tensor = allocator->allocate_tensor<float>({6, 6});
//
//     // 2 is the number of dimensions of this tensor
//     auto accessor = tensor->accessor<2>();
//     accessor[5][3] = 1.0;
//
// Accessors are implemented using recursive templates and work by computing the correct offsets into the
// tensor's underlying buffer. To do this, they keep track of 4 things: the dimensions of the tensor, the strides
// of the tensor, the current accessor's starting offset into the tensor's buffer, and a pointer to the tensor's
// underlying buffer.
//
// Note that TensorAccessors are only valid as long as the accessed NeuropodTensor is still in scope
template <typename Container, size_t N>
class TensorAccessor
{
private:
    // `data_` is a container that supports the subscript operator ([]) to access data.
    // For numeric tensor types, this is usually a pointer to a tensor's underlying buffer
    Container data_;

    // `dims_` points to an array containing the dimensions of the tensor
    const int64_t *dims_;

    // `strides_` points to an array containing the strides of a tensor
    const int64_t *strides_;

    // `offset_` is the offset into `data_` where this accessor starts
    const int64_t offset_;

public:
    TensorAccessor(Container data, const int64_t *dims, const int64_t *strides, int64_t offset = 0)
        : data_(data), dims_(dims), strides_(strides), offset_(offset)
    {
    }

    // Indexing into a N dimensional accessor returns an N - 1 dimensional accessor
    const TensorAccessor<Container, N - 1> operator[](int64_t i) const
    {
        // This operator returns a TensorAccessor that accesses an `N - 1` dimensional tensor at index `i`
        // of this TensorAccessor. To do this, we compute the correct offsets into `data_` and pass along the last
        // `N - 1` elements in `dims_` and `strides_` to the new accessor. For example:
        //
        //     auto tensor = allocator->allocate_tensor<float>({3, 5});
        //
        //     // Not using `auto` for clarity on types
        //     TensorAccessor<float *, 2> accessor = tensor->accessor();
        //
        //     // In this accessor:
        //     // `data_` points to the tensor's underlying buffer
        //     // `dims_` points to an array containing 3, 5
        //     // `strides_` points to an array containing 5, 1
        //     // `offset_` is 0
        //     // This means that any indexing into this accessor indexes into the underlying buffer
        //     // starting at `0` with a stride of `5`
        //
        //     // Since `N` (in the template args) is > 1, this accessor returns another accessor when indexed into
        //     TensorAccessor<float *, 1> subaccessor = accessor[2];
        //
        //     // In this accessor:
        //     // `data_` points to the tensor's underlying buffer
        //     // `dims_` points to an array containing 5
        //     // `strides_` points to an array containing 1
        //     // `offset_` is 10
        //     // This means that any indexing into this accessor indexes into the underlying buffer
        //     // starting at `10` with a stride of `1`
        //
        //     // This is equivalent to an index of 11 in the underlying buffer (or the 12th item)
        //     float item = subaccessor[1];
        //
        //     // Same as this
        //     float same_item = accessor[2][1];
        //
        //     // Same as this
        //     float also_same = *(tensor->get_raw_data_ptr() + 11);
        //
        return TensorAccessor<Container, N - 1>(data_, dims_ + 1, strides_ + 1, offset_ + strides_[0] * i);
    }

    // begin and end (to support range-based for loops)
    auto begin() const -> decltype(get_iterator(this)) { return get_iterator(this); }
    auto end() const -> decltype(get_iterator(this, dims_[0])) { return get_iterator(this, dims_[0]); }
};

// Specialization for base case
template <typename Container>
class TensorAccessor<Container, 1>
{
private:
    Container      data_;
    const int64_t *dims_;
    const int64_t *strides_;
    const int64_t  offset_;

public:
    TensorAccessor(Container data, const int64_t *dims, const int64_t *strides, int64_t offset = 0)
        : data_(data), dims_(dims), strides_(strides), offset_(offset)
    {
    }

    // Data access
    auto operator[](int64_t i) const -> decltype(data_[offset_ + i]) { return data_[offset_ + i]; }

    // begin and end (to support range-based for loops)
    auto begin() const -> decltype(get_iterator(this)) { return get_iterator(this); }
    auto end() const -> decltype(get_iterator(this, dims_[0])) { return get_iterator(this, dims_[0]); }
};

// A struct that wraps a TensorAccessor along with dims and strides
// This is generally used for "viewing" a tensor with dims other than the original.
// The restrictions of TensorAccessor above apply here, but additionally, any returned
// TensorAccessors are only valid as long as the TensorView is still in scope
template <typename Container, size_t N>
class TensorView
{
private:
    std::vector<int64_t>         dims_;
    std::vector<int64_t>         strides_;
    TensorAccessor<Container, N> accessor_;

public:
    TensorView(Container data, std::vector<int64_t> dims, std::vector<int64_t> strides)
        : dims_(std::move(dims)), strides_(std::move(strides)), accessor_(data, dims_.data(), strides_.data())
    {
    }

    decltype(auto) operator[](int64_t i) const { return accessor_[i]; }

    auto begin() const { return accessor_.begin(); }
    auto end() const { return accessor_.end(); }

    // To get direct access to an accessor for the entire view
    auto accessor() const { return accessor_; }
};

} // namespace neuropod

namespace std
{

// Need to specialize `std::iterator_traits` for `AccessorIterator`
template <typename T>
struct iterator_traits<neuropod::AccessorIterator<T>>
{
    using value_type = typename neuropod::AccessorIterator<T>::value_type;
};

} // namespace std
