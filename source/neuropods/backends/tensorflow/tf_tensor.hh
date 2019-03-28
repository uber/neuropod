//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <vector>

#include <tensorflow/c/c_api.h>

#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/backends/tensorflow/type_utils.hh"

namespace neuropods
{

namespace
{

// Get the shape of a TF tensor
std::vector<int64_t> get_shape(TF_Tensor *tensor)
{
    const int            dims = TF_NumDims(tensor);
    std::vector<int64_t> out;

    for (int curr_dim = 0; curr_dim < dims; curr_dim++)
    {
        out.push_back(TF_Dim(tensor, curr_dim));
    }

    return out;
}

void deallocator(void * data, size_t len, void * handle)
{
    // The tensor is being deallocated, run the deleter that the user provided
    run_deleter(handle);
}

} // namespace


// This class is internal to neuropods and should not be exposed
// to users
template<typename T>
class TensorflowNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<TF_Tensor *>
{
public:
    // Allocate a TF tensor
    TensorflowNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<T>(name, dims),
          tensor(TF_AllocateTensor(get_tf_type_from_neuropod_type(this->get_tensor_type()),
                                   dims.data(),
                                   dims.size(),
                                   this->get_num_elements() * sizeof(T)))
    {
    }

    // Wrap existing memory
    TensorflowNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims, void * data, const Deleter &deleter)
        : TypedNeuropodTensor<T>(name, dims),
          tensor(TF_NewTensor(get_tf_type_from_neuropod_type(this->get_tensor_type()),
                              dims.data(),
                              dims.size(),
                              data,
                              this->get_num_elements() * sizeof(T),
                              deallocator,
                              register_deleter(deleter, data)
                              ))
    {
    }

    // Wrap an existing TF tensor
    TensorflowNeuropodTensor(const std::string &name, TF_Tensor *tensor)
        : TypedNeuropodTensor<T>(name, get_shape(tensor)),
          tensor(tensor)
    {
    }

    ~TensorflowNeuropodTensor()
    {
        if (tensor != nullptr)
        {
            TF_DeleteTensor(tensor);
        }
    }

    // Get a pointer to the underlying data
    T *get_raw_data_ptr() { return static_cast<T *>(TF_TensorData(tensor)); }

    // Get a pointer to the underlying data
    const T *get_raw_data_ptr() const { return static_cast<T *>(TF_TensorData(tensor)); }

    TF_Tensor *get_native_data() { return tensor; }

    // The underlying TF tensor
    TF_Tensor *tensor;
};


// Specialization for strings
template <>
class TensorflowNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>,
                                              public NativeDataContainer<TF_Tensor *>
{
public:
    // Allocate a TF tensor
    TensorflowNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<std::string>(name, dims)
    {
    }

    // Wrap an existing TF tensor
    TensorflowNeuropodTensor(const std::string &name, TF_Tensor *tensor)
        : TypedNeuropodTensor<std::string>(name, get_shape(tensor)), tensor(tensor)
    {
    }

    ~TensorflowNeuropodTensor()
    {
        if (tensor != nullptr)
        {
            TF_DeleteTensor(tensor);
        }
    }

    void set(const std::vector<std::string> &data)
    {
        // The format of TF string tensors is described here:
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h#L214

        // Make sure that the size of the provided data matches the number of
        // elements in the tensor
        if (data.size() != this->get_num_elements())
        {
            NEUROPOD_ERROR("Size of provided vector does not match the number "
                "of elements in the tensor");
        }

        // Get the size of the index
        const size_t index_size = sizeof(uint64_t) * data.size();

        // Compute the total size of the tensor
        size_t total_bytes = index_size;
        for (const auto &item : data)
        {
            total_bytes += TF_StringEncodedSize(item.length());
        }

        // Allocate the tensor
        const auto dims = get_dims();
        tensor          = TF_AllocateTensor(TF_STRING, dims.data(), dims.size(), total_bytes);

        // Get pointers to the data
        void *    tensor_data_ptr = TF_TensorData(tensor);
        uint64_t *index_ptr       = static_cast<uint64_t *>(tensor_data_ptr);
        char *    data_ptr        = static_cast<char *>(tensor_data_ptr) + index_size;

        // Set the data
        TF_Status *status   = TF_NewStatus();
        size_t     counter  = 0;
        uint64_t   position = 0;
        for (const auto &item : data)
        {
            // Get the lengths and set the index entry
            const auto len         = item.length();
            const auto encoded_len = TF_StringEncodedSize(len);
            index_ptr[counter++]   = position;

            // Encode the string
            TF_StringEncode(item.c_str(), len, data_ptr + position, encoded_len, status);

            // Check status
            if (TF_GetCode(status) != TF_OK)
            {
                NEUROPOD_ERROR("Tensorflow error: " << TF_Message(status));
            }

            // Move the pointer forward
            position += encoded_len;
        }

        // Delete the status
        TF_DeleteStatus(status);
    }

    std::vector<std::string> get_data_as_vector()
    {
        // The format of TF string tensors is described here:
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h#L214
        std::vector<std::string> out;

        // Get pointers to the data and sizes
        const size_t     tensor_size     = TF_TensorByteSize(tensor);
        const size_t     numel           = get_num_elements();
        void * const     tensor_data_ptr = TF_TensorData(tensor);
        uint64_t * const index_ptr       = static_cast<uint64_t *>(tensor_data_ptr);
        const size_t     index_size      = sizeof(uint64_t) * numel;
        char * const     data_ptr        = static_cast<char *>(tensor_data_ptr) + index_size;

        TF_Status *status = TF_NewStatus();
        for (size_t i = 0; i < numel; i++)
        {
            const uint64_t position = index_ptr[i];
            const char *   dst;
            size_t         buf_len;

            // Decode the string
            TF_StringDecode(data_ptr + position, tensor_size - position - index_size, &dst, &buf_len, status);

            // Check status
            if (TF_GetCode(status) != TF_OK)
            {
                NEUROPOD_ERROR("Tensorflow error: " << TF_Message(status));
            }

            // Add the string
            out.emplace_back(dst, buf_len);
        }

        // Delete the status
        TF_DeleteStatus(status);

        return out;
    }

    TF_Tensor *get_native_data() { return tensor; }

    // The underlying TF tensor
    TF_Tensor *tensor;
};

} // namespace neuropods
