//
// Uber, Inc. (c) 2020
//

#pragma once

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/internal/deleter.hh"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace neuropod
{

template <typename T>
class GenericNeuropodTensor : public TypedNeuropodTensor<T>
{
private:
    // A pointer to the data contained in the tensor
    void *data_;

    // A deleter to free the underlying memory
    void *deleter_handle_;

public:
    GenericNeuropodTensor(const std::vector<int64_t> &dims) : TypedNeuropodTensor<T>(dims)
    {
        data_           = malloc(this->get_num_elements() * sizeof(T));
        deleter_handle_ = register_deleter([](void *data) { free(data); }, data_);
    }

    // Wrap existing memory
    GenericNeuropodTensor(const std::vector<int64_t> &dims, void *data, const Deleter &deleter)
        : TypedNeuropodTensor<T>(dims)
    {
        data_           = data;
        deleter_handle_ = register_deleter(deleter, data);
    }

    ~GenericNeuropodTensor() { run_deleter(deleter_handle_); }

protected:
    // Get a pointer to the underlying data
    void *get_untyped_data_ptr() { return data_; }

    const void *get_untyped_data_ptr() const { return data_; }
};

// Specialization for strings
template <>
class GenericNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>
{
private:
    // The data contained in the tensor
    std::vector<std::string> data_;

public:
    GenericNeuropodTensor(const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<std::string>(dims), data_(this->get_num_elements())
    {
    }

    ~GenericNeuropodTensor() = default;

    void copy_from(const std::vector<std::string> &data) { data_ = data; }

protected:
    std::string get(size_t index) const { return data_[index]; }

    void set(size_t index, const std::string &value) { data_[index] = value; }
};

// Get a `NeuropodTensorAllocator` that creates `GenericNeuropodTensor`s
std::unique_ptr<NeuropodTensorAllocator> get_generic_tensor_allocator();

} // namespace neuropod
