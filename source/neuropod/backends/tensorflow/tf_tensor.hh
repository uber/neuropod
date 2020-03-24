//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropod/backends/tensorflow/type_utils.hh"
#include "neuropod/internal/deleter.hh"
#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/logging.hh"
#include "neuropod/internal/neuropod_tensor.hh"
#include "tensorflow/core/framework/tensor.h"

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
void create_tensor_from_existing_memory(const std::vector<int64_t> &         dims,
                                        void *                               data,
                                        const Deleter &                      deleter,
                                        size_t                               data_size_bytes,
                                        tensorflow::DataType                 type,
                                        std::shared_ptr<tensorflow::Tensor> &tensor);

} // namespace detail

// A container for a tensorflow Tensor
class SealedTensorflowTensor : public SealedNeuropodTensor
{
public:
    std::shared_ptr<tensorflow::Tensor> value;
};

template <typename T>
class TensorflowNeuropodTensor : public TypedNeuropodTensor<T>
{
private:
    std::shared_ptr<tensorflow::Tensor> tensor_;

public:
    // Allocate a TF tensor
    TensorflowNeuropodTensor(const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<T>(dims),
          tensor_(std::make_shared<tensorflow::Tensor>(get_tf_type_from_neuropod_type(this->get_tensor_type()),
                                                       detail::get_tf_shape(dims)))
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
                                                   tensor_);
    }

    // Wrap an existing TF tensor
    TensorflowNeuropodTensor(tensorflow::Tensor tensor)
        : TypedNeuropodTensor<T>(detail::get_dims(tensor)),
          tensor_(std::make_shared<tensorflow::Tensor>(std::move(tensor)))
    {
    }

    ~TensorflowNeuropodTensor() = default;

protected:
    // Get a pointer to the underlying data
    void *get_untyped_data_ptr()
    {
        // TODO(vip): make sure this tensor is contiguous
        return const_cast<char *>(tensor_->tensor_data().data());
    }

    // Get a pointer to the underlying data
    const void *get_untyped_data_ptr() const
    {
        // TODO(vip): make sure this tensor is contiguous
        return tensor_->tensor_data().data();
    }

    std::shared_ptr<SealedNeuropodTensor> seal(NeuropodDevice device)
    {
        auto out = std::make_shared<SealedTensorflowTensor>();

        // TODO(vip): Move to the correct device
        // TODO(vip): std::move
        out->value = tensor_;

        return out;
    }
};

// Specialization for strings
template <>
class TensorflowNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>
{
private:
    std::shared_ptr<tensorflow::Tensor> tensor_;

public:
    // Allocate a TF tensor
    TensorflowNeuropodTensor(const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<std::string>(dims),
          tensor_(std::make_shared<tensorflow::Tensor>(tensorflow::DT_STRING, detail::get_tf_shape(dims)))
    {
    }

    // Wrap an existing TF tensor
    TensorflowNeuropodTensor(tensorflow::Tensor tensor)
        : TypedNeuropodTensor<std::string>(detail::get_dims(tensor)),
          tensor_(std::make_shared<tensorflow::Tensor>(std::move(tensor)))
    {
    }

    ~TensorflowNeuropodTensor() = default;

    void set(const std::vector<std::string> &data)
    {
        auto flat = tensor_->flat<std::string>();
        if (data.size() != flat.size())
        {
            NEUROPOD_ERROR("Supplied vector size ({}) does not match size of tensor ({})", data.size(), flat.size());
        }

        for (int i = 0; i < data.size(); i++)
        {
            flat(i) = data[i];
        }
    }

protected:
    const std::string operator[](size_t index) const { return tensor_->flat<std::string>()(index); }

    std::shared_ptr<SealedNeuropodTensor> seal(NeuropodDevice device)
    {
        auto out = std::make_shared<SealedTensorflowTensor>();

        // TODO(vip): std::move
        out->value = tensor_;

        return out;
    }
};

} // namespace neuropod
