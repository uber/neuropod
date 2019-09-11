//
// Uber, Inc. (c) 2019
//

#pragma once

namespace neuropods
{

// Inspired by https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/TensorAccessor.h
template <typename T, size_t N>
class TensorAccessor
{
private:
    T *            data_;
    const int64_t *strides_;

public:
    TensorAccessor(T *data, const int64_t *strides) : data_(data), strides_(strides) {}

    TensorAccessor<T, N - 1> operator[](int64_t i)
    {
        return TensorAccessor<T, N - 1>(this->data_ + this->strides_[0] * i, this->strides_ + 1);
    }

    const TensorAccessor<T, N - 1> operator[](int64_t i) const
    {
        return TensorAccessor<T, N - 1>(this->data_ + this->strides_[0] * i, this->strides_ + 1);
    }
};

// Specialization for base case
template <typename T>
class TensorAccessor<T, 1>
{
private:
    T *            data_;
    const int64_t *strides_;

public:
    TensorAccessor(T *data, const int64_t *strides) : data_(data), strides_(strides) {}

    T &operator[](int64_t i) { return this->data_[i]; }

    const T &operator[](int64_t i) const { return this->data_[i]; }
};

} // namespace neuropods
