//
// Uber, Inc. (c) 2019
//

#pragma once

namespace neuropods
{

// Forward declare TensorAccessor
template <typename T, size_t N>
class TensorAccessor;

// An iterator type used by the accessors
template <typename T, size_t N>
class AccessorIterator
{
private:
    TensorAccessor<T, N> *accessor_;
    size_t                index_;

public:
    AccessorIterator(TensorAccessor<T, N> *accessor, size_t index = 0) : accessor_(accessor), index_(index) {}

    AccessorIterator<T, N> &operator++()
    {
        index_++;
        return *this;
    }

    TensorAccessor<T, N - 1> operator*() { return (*accessor_)[index_]; }

    bool operator!=(const AccessorIterator<T, N> &other)
    {
        return index_ != other.index_ || accessor_ != other.accessor_;
    }
};

// A const iterator type used by the const accessors
template <typename T, size_t N>
class AccessorConstIterator
{
private:
    const TensorAccessor<T, N> *accessor_;
    size_t                      index_;

public:
    AccessorConstIterator(const TensorAccessor<T, N> *accessor, size_t index = 0) : accessor_(accessor), index_(index)
    {
    }

    AccessorConstIterator<T, N> &operator++()
    {
        index_++;
        return *this;
    }

    const TensorAccessor<T, N - 1> operator*() { return (*accessor_)[index_]; }

    bool operator!=(const AccessorConstIterator<T, N> &other)
    {
        return index_ != other.index_ || accessor_ != other.accessor_;
    }
};

// Inspired by https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/TensorAccessor.h
template <typename T, size_t N>
class TensorAccessor
{
private:
    T *            data_;
    const int64_t *dims_;
    const int64_t *strides_;

public:
    TensorAccessor(T *data, const int64_t *dims, const int64_t *strides) : data_(data), dims_(dims), strides_(strides)
    {
    }

    TensorAccessor<T, N - 1> operator[](size_t i)
    {
        return TensorAccessor<T, N - 1>(this->data_ + this->strides_[0] * i, this->dims_ + 1, this->strides_ + 1);
    }

    const TensorAccessor<T, N - 1> operator[](size_t i) const
    {
        return TensorAccessor<T, N - 1>(this->data_ + this->strides_[0] * i, this->dims_ + 1, this->strides_ + 1);
    }

    AccessorIterator<T, N> begin() { return AccessorIterator<T, N>(this); }

    AccessorConstIterator<T, N> begin() const { return AccessorConstIterator<T, N>(this); }

    AccessorIterator<T, N> end() { return AccessorIterator<T, N>(this, dims_[0]); }

    AccessorConstIterator<T, N> end() const { return AccessorConstIterator<T, N>(this, dims_[0]); }
};

// Specialization for base case
template <typename T>
class TensorAccessor<T, 1>
{
private:
    T *            data_;
    const int64_t *dims_;
    const int64_t *strides_;

public:
    TensorAccessor(T *data, const int64_t *dims, const int64_t *strides) : data_(data), dims_(dims), strides_(strides)
    {
    }

    T &operator[](size_t i) { return this->data_[i]; }

    const T &operator[](size_t i) const { return this->data_[i]; }

    const T *begin() const { return &this->data_[0]; }

    T *begin() { return &this->data_[0]; }

    const T *end() const { return &this->data_[this->dims_[0]]; }

    T *end() { return &this->data_[this->dims_[0]]; }
};

} // namespace neuropods
