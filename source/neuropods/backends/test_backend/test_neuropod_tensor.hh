//
// Uber, Inc. (c) 2018
//

#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/neuropod_tensor.hh"

namespace neuropods
{


// This is used along with the TestNeuropodBackend in tests
template <typename T>
class TestNeuropodTensor : public TypedNeuropodTensor<T>
{
private:
    // A pointer to the data contained in the tensor
    void *data_;

    // A deleter to free the underlying memory
    void *deleter_handle_;

public:
    TestNeuropodTensor(const std::vector<int64_t> &dims) : TypedNeuropodTensor<T>(dims)
    {
        data_ = malloc(this->get_num_elements() * sizeof(T));
        deleter_handle_ = register_deleter([](void * data) { free(data); }, data_);
    }

    // Wrap existing memory
    TestNeuropodTensor(const std::vector<int64_t> &dims, void * data, const Deleter &deleter)
        : TypedNeuropodTensor<T>(dims)
    {
        data_ = data;
        deleter_handle_ = register_deleter(deleter, data);
    }

    ~TestNeuropodTensor() { run_deleter(deleter_handle_); }

    // Get a pointer to the underlying data
    T *get_raw_data_ptr() { return static_cast<T *>(data_); }

    const T *get_raw_data_ptr() const { return static_cast<T *>(data_); }
};

// Specialization for strings
template <>
class TestNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>
{
private:
    // The data contained in the tensor
    std::vector<std::string> data_;

public:
    TestNeuropodTensor(const std::vector<int64_t> &dims) : TypedNeuropodTensor<std::string>(dims)
    {
    }

    ~TestNeuropodTensor() = default;

    void set(const std::vector<std::string> &data)
    {
        data_ = data;
    }

    std::vector<std::string> get_data_as_vector()
    {
        return data_;
    }
};

} // namespace neuropods
