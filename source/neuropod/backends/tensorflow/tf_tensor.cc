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

#include "neuropod/backends/tensorflow/tf_tensor.hh"

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow
{

// Before TF 1.15, the `Tensor` constructor with an existing TensorBuffer was
// private and only usable by friend classes (including the `TensorCApi` class).
// In order for us to create a Tensor using an existing buffer, we must use one of these
// friend classes.
// Much of the code below was inspired by (or is from) the TensorFlow C API.
class TensorCApi
{
public:
    // NOLINTNEXTLINE(readability-identifier-naming): Friend function declaration within TF names it this way
    static Tensor MakeTensor(DataType type, const TensorShape &shape, TensorBuffer *buf)
    {
        return Tensor(type, shape, buf);
    }
};

} // namespace tensorflow

namespace neuropod::detail
{

namespace
{

// Based on TF_ManagedBuffer in https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_tensor.cc
class NeuropodTensorBuffer : public tensorflow::TensorBuffer
{
private:
#if TF_MAJOR_VERSION <= 1 && TF_MINOR_VERSION <= 12
    void *data_;
#endif
    const size_t len_;
    void *       deleter_handle_;

public:
    NeuropodTensorBuffer(void *data, size_t len, void *deleter_handle)
        :
#if (TF_MAJOR_VERSION <= 1 && TF_MINOR_VERSION > 12) || TF_MAJOR_VERSION > 1
          TensorBuffer(data),
#else
          data_(data),
#endif
          len_(len),
          deleter_handle_(deleter_handle)
    {
    }

    ~NeuropodTensorBuffer() override
    {
        // The tensor is being deallocated, run the deleter that the user provided
        run_deleter(deleter_handle_);
    }

#if TF_MAJOR_VERSION <= 1 && TF_MINOR_VERSION <= 12
    void *data() const override { return data_; }
#endif
    size_t        size() const override { return len_; }
    TensorBuffer *root_buffer() override { return this; }

    void FillAllocationDescription(tensorflow::AllocationDescription *proto) const override
    {
        auto rb = static_cast<tensorflow::int64>(size());
        proto->set_requested_bytes(rb);
        proto->set_allocator_name(tensorflow::cpu_allocator()->Name());
    }

    // Prevents input forwarding from mutating this buffer.
    bool OwnsMemory() const override { return false; }
};

// TODO(vip): Use std::align
void *get_next_aligned_offset(void *base)
{
    // We want to find an offset such that the data will be 64 byte aligned
    auto   base_address   = reinterpret_cast<uint64_t>(base);
    size_t aligned_offset = 64 - (base_address % 64);
    return reinterpret_cast<void *>(base_address + aligned_offset);
}

} // namespace

// Convert shapes
tensorflow::TensorShape get_tf_shape(const std::vector<int64_t> &dims)
{
    tensorflow::TensorShape shape;
    for (const auto dim : dims)
    {
        shape.AddDim(dim);
    }

    return shape;
}

// Convert shapes
std::vector<int64_t> get_dims(const tensorflow::Tensor &tensor)
{
    auto                 num_dims = static_cast<size_t>(tensor.dims());
    std::vector<int64_t> shape(num_dims);
    for (size_t i = 0; i < num_dims; i++)
    {
        shape[i] = tensor.dim_size(static_cast<int>(i));
    }

    return shape;
}

void create_tensor_from_existing_memory(const std::vector<int64_t> &dims,
                                        void *                      data,
                                        const Deleter &             deleter,
                                        size_t                      data_size_bytes,
                                        tensorflow::DataType        type,
                                        tensorflow::Tensor &        tensor,
                                        tensorflow::TensorBuffer *& buf)
{
    auto deleter_handle = register_deleter(deleter, data);
    if (reinterpret_cast<intptr_t>(data) % 64 != 0)
    {
        SPDLOG_WARN("In order to wrap data, TensorFlow expects data to be 64 byte aligned! Making a copy...");

        // Copy the data
        void *copy_buffer = malloc(data_size_bytes + 64);
        void *aligned     = detail::get_next_aligned_offset(copy_buffer);

        memcpy(aligned, data, data_size_bytes);

        // Run the deleter on the original data since we no longer need it
        run_deleter(deleter_handle);

        // Register a new deleter and update the buffer
        deleter_handle = register_deleter([](void *to_free) { free(to_free); }, copy_buffer);
        data           = aligned;
    }

    buf    = new NeuropodTensorBuffer(data, data_size_bytes, deleter_handle);
    tensor = tensorflow::TensorCApi::MakeTensor(type, detail::get_tf_shape(dims), buf);
}

} // namespace neuropod::detail
