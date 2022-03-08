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

#include "neuropod/internal/deleter.hh"
#include "neuropod/internal/neuropod_tensor.hh"
#include "neuropod/internal/tensor_types.hh"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace neuropod
{

// An base class used to allocate tensors
class NeuropodTensorAllocator
{
public:
    virtual ~NeuropodTensorAllocator() {}

    // Allocate a tensor of a specific type
    virtual std::unique_ptr<NeuropodTensor> allocate_tensor(const std::vector<int64_t> &input_dims,
                                                            TensorType                  tensor_type) = 0;

    // Allocate a tensor of a specific type and wrap existing memory.
    // Note: Some backends may have specific alignment requirements (e.g. tensorflow).
    // To support all the built-in backends, `data` should be aligned to 64 bytes.
    // `deleter` will be called with a pointer to `data` when the tensor is
    // deallocated
    virtual std::unique_ptr<NeuropodTensor> tensor_from_memory(const std::vector<int64_t> &input_dims,
                                                               TensorType                  tensor_type,
                                                               void *                      data,
                                                               const Deleter &             deleter) = 0;

    // Templated version of `allocate_tensor`
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> allocate_tensor(const std::vector<int64_t> &input_dims)
    {
        std::shared_ptr<NeuropodTensor> tensor = this->allocate_tensor(input_dims, get_tensor_type_from_cpp<T>());

        return std::dynamic_pointer_cast<TypedNeuropodTensor<T>>(tensor);
    }

    // Templated version of `tensor_from_memory`
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> tensor_from_memory(const std::vector<int64_t> &input_dims,
                                                               T *                         data,
                                                               const Deleter &             deleter)
    {
        std::shared_ptr<NeuropodTensor> tensor =
            this->tensor_from_memory(input_dims, get_tensor_type_from_cpp<T>(), data, deleter);

        return std::dynamic_pointer_cast<TypedNeuropodTensor<T>>(tensor);
    }

    // Returns a tensor of type `T` and shape `input_dims` filled with `fill_value`
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> full(const std::vector<int64_t> &input_dims, T fill_value)
    {
        auto tensor = allocate_tensor<T>(input_dims);
        std::fill_n(tensor->get_raw_data_ptr(), tensor->get_num_elements(), fill_value);
        return tensor;
    }

    // Returns a tensor of type `T` and shape `input_dims` filled with zeros
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> zeros(const std::vector<int64_t> &input_dims)
    {
        return full<T>(input_dims, 0);
    }

    // Returns a tensor of type `T` and shape `input_dims` filled with ones
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> ones(const std::vector<int64_t> &input_dims)
    {
        return full<T>(input_dims, 1);
    }

    // Returns a 1D tensor of type `T` containing a sequence of numbers starting at `start`
    // with a step size of `step`. The tensor has the following shape: (`ceil((end - start) / step)`)
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> arange(T start, T end, T step = 1)
    {
        // Note: This is intended to be used in tests and therefore is not optimized for performance

        // Compute the size and allocate a tensor
        int64_t size   = std::ceil((end - start) / static_cast<double>(step));
        auto    tensor = allocate_tensor<T>({size});

        // Fill the tensor
        auto       data_ptr = tensor->get_raw_data_ptr();
        const auto numel    = tensor->get_num_elements();
        for (int i = 0, val = 0; i < numel; i++, val += step)
        {
            data_ptr[i] = start + val;
        }

        return tensor;
    }

    // Returns a 1D tensor of type `T` containing a sequence of numbers starting at `0` and
    // ending at `end - 1` with a step size of `1`. The tensor has the following shape: (`ceil(end)`)
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> arange(T end)
    {
        return arange<T>(0, end);
    }

    // Returns an identity matrix of type `T` and shape (`M`, `N`). This matrix has ones on the diagonal and
    // zeros everywhere else.
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> eye(int64_t M, int64_t N)
    {
        // Note: This is intended to be used in tests and therefore is not optimized for performance

        auto smallest_dim = std::min(M, N);
        auto tensor       = zeros<T>({M, N});
        auto accessor     = tensor->template accessor<2>();

        for (int i = 0; i < smallest_dim; i++)
        {
            accessor[i][i] = 1;
        }

        return tensor;
    }

    // Returns a tensor of type `T` and shape `input_dims` filled with random numbers from a normal
    // distribution with mean `mean` and standard deviation `stddev`.
    template <typename T>
    std::shared_ptr<TypedNeuropodTensor<T>> randn(const std::vector<int64_t> &input_dims, T mean = 0, T stddev = 1)
    {
        // Note: This is intended to be used in tests and therefore is not optimized for performance

        // Allocate a tensor
        auto tensor = allocate_tensor<T>(input_dims);

        // Setup random number generation
        std::random_device          rd;
        std::mt19937                gen(rd());
        std::normal_distribution<T> d(mean, stddev);

        // Fill the tensor with random numbers
        auto       data_ptr = tensor->get_raw_data_ptr();
        const auto numel    = tensor->get_num_elements();
        for (int i = 0; i < numel; i++)
        {
            data_ptr[i] = d(gen);
        }

        return tensor;
    }
};

// A default allocator
template <template <class> class TensorImpl>
class DefaultTensorAllocator : public NeuropodTensorAllocator
{
public:
    std::unique_ptr<NeuropodTensor> allocate_tensor(const std::vector<int64_t> &input_dims, TensorType tensor_type)
    {
        return make_tensor<TensorImpl>(tensor_type, input_dims);
    }

    std::unique_ptr<NeuropodTensor> tensor_from_memory(const std::vector<int64_t> &input_dims,
                                                       TensorType                  tensor_type,
                                                       void *                      data,
                                                       const Deleter &             deleter)
    {
        return make_tensor_no_string<TensorImpl>(tensor_type, input_dims, data, deleter);
    }
};

} // namespace neuropod
