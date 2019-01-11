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
};

} // namespace neuropods
