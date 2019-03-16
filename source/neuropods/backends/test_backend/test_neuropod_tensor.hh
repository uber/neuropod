//
// Uber, Inc. (c) 2018
//

#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <vector>

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

public:
    TestNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims) : TypedNeuropodTensor<T>(name, dims)
    {
        data_ = malloc(this->get_num_elements() * sizeof(T));
    }

    ~TestNeuropodTensor() { free(data_); }

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
    TestNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims) : TypedNeuropodTensor<std::string>(name, dims)
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
