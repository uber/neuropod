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

#include "neuropod/backends/tensorflow/type_utils.hh"
#include "neuropod/internal/deleter.hh"
#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/logging.hh"
#include "neuropod/internal/neuropod_tensor.hh"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/version.h"

#include <string>
#include <vector>

namespace neuropod
{

namespace detail
{

// Convert shapes
tensorflow::TensorShape get_tf_shape(const std::vector<int64_t> &dims);

// Convert shapes
std::vector<int64_t> get_dims(const tensorflow::Tensor &tensor);

// Create a TF tensor from existing memory
void create_tensor_from_existing_memory(const std::vector<int64_t> &dims,
                                        void *                      data,
                                        const Deleter &             deleter,
                                        size_t                      data_size_bytes,
                                        tensorflow::DataType        type,
                                        tensorflow::Tensor &        tensor,
                                        tensorflow::TensorBuffer *& buf);

} // namespace detail

template <typename T>
class TensorflowNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<tensorflow::Tensor &>
{
private:
    tensorflow::Tensor        tensor_;
    tensorflow::TensorBuffer *buf_ = nullptr;

public:
    // Allocate a TF tensor
    TensorflowNeuropodTensor(const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<T>(dims),
          tensor_(get_tf_type_from_neuropod_type(this->get_tensor_type()), detail::get_tf_shape(dims))
    {
    }

    // Wrap existing memory
    // This data should be 64 byte aligned
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/allocator.h#L84
    TensorflowNeuropodTensor(const std::vector<int64_t> &dims, void *data, const Deleter &deleter)
        : TypedNeuropodTensor<T>(dims)
    {
        detail::create_tensor_from_existing_memory(dims,
                                                   data,
                                                   deleter,
                                                   this->get_num_elements() * sizeof(T),
                                                   get_tf_type_from_neuropod_type(this->get_tensor_type()),
                                                   tensor_,
                                                   buf_);
    }

    // Wrap an existing TF tensor
    TensorflowNeuropodTensor(tensorflow::Tensor tensor)
        : TypedNeuropodTensor<T>(detail::get_dims(tensor)), tensor_(std::move(tensor))
    {
    }

    ~TensorflowNeuropodTensor()
    {
        if (buf_ != nullptr)
        {
            // Reset the tensor
            tensor_ = tensorflow::Tensor();

            // Unref the buffer
            buf_->Unref();
        }
    }

    tensorflow::Tensor &get_native_data() { return tensor_; }

protected:
    // Get a pointer to the underlying data
    void *get_untyped_data_ptr()
    {
        // TODO(vip): make sure this tensor is contiguous
        auto buf = tensorflow::DMAHelper::buffer(&tensor_);
        if (buf != nullptr)
        {
            return buf->data();
        }

        return nullptr;
    }

    // Get a pointer to the underlying data
    const void *get_untyped_data_ptr() const
    {
        // TODO(vip): make sure this tensor is contiguous
        auto buf = tensorflow::DMAHelper::buffer(&tensor_);
        if (buf != nullptr)
        {
            return buf->data();
        }

        return nullptr;
    }
};

// Specialization for strings
template <>
class TensorflowNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>,
                                              public NativeDataContainer<tensorflow::Tensor &>
{
private:
    tensorflow::Tensor tensor_;

#if TF_MAJOR_VERSION > 1
    using StringType = tensorflow::tstring;
#else
    using StringType = std::string;
#endif

public:
    // Allocate a TF tensor
    TensorflowNeuropodTensor(const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<std::string>(dims), tensor_(tensorflow::DT_STRING, detail::get_tf_shape(dims))
    {
    }

    // Wrap an existing TF tensor
    TensorflowNeuropodTensor(tensorflow::Tensor tensor)
        : TypedNeuropodTensor<std::string>(detail::get_dims(tensor)), tensor_(std::move(tensor))
    {
    }

    ~TensorflowNeuropodTensor() = default;

    void copy_from(const std::vector<std::string> &data)
    {
        auto flat = tensor_.flat<StringType>();
        if (data.size() != static_cast<size_t>(flat.size()))
        {
            NEUROPOD_ERROR("Supplied vector size ({}) does not match size of tensor ({})", data.size(), flat.size());
        }

        for (size_t i = 0; i < data.size(); i++)
        {
            flat(static_cast<long>(i)) = data[i];
        }
    }

    tensorflow::Tensor &get_native_data() { return tensor_; }

protected:
    std::string get(size_t index) const { return tensor_.flat<StringType>()(static_cast<long>(index)); }

    void set(size_t index, const std::string &value)
    {
        auto flat                      = tensor_.flat<StringType>();
        flat(static_cast<long>(index)) = value;
    }
};

} // namespace neuropod
