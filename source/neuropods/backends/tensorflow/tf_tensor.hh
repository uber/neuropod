//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/backends/tensorflow/type_utils.hh"
#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/logging.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "tensorflow/core/framework/tensor.h"

#include <string>
#include <vector>

namespace neuropods
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
        return const_cast<char *>(tensor_.tensor_data().data());
    }

    // Get a pointer to the underlying data
    const void *get_untyped_data_ptr() const
    {
        // TODO(vip): make sure this tensor is contiguous
        return tensor_.tensor_data().data();
    }
};

// Specialization for strings
template <>
class TensorflowNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>,
                                              public NativeDataContainer<tensorflow::Tensor &>
{
private:
    tensorflow::Tensor tensor_;

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

    void set(const std::vector<std::string> &data)
    {
        auto flat = tensor_.flat<std::string>();
        if (data.size() != flat.size())
        {
            NEUROPOD_ERROR("Supplied vector size (" << data.size() << ") does not match size of tensor (" << flat.size()
                                                    << ")");
        }

        for (int i = 0; i < data.size(); i++)
        {
            flat(i) = data[i];
        }
    }

    std::vector<std::string> get_data_as_vector() const
    {
        auto                     flat = tensor_.flat<std::string>();
        std::vector<std::string> out(flat.size());
        for (int i = 0; i < out.size(); i++)
        {
            out[i] = flat(i);
        }

        return out;
    }

    tensorflow::Tensor &get_native_data() { return tensor_; }
};

} // namespace neuropods
