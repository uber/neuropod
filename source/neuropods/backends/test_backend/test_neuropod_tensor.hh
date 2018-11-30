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
class TestNeuropodTensor : public NeuropodTensor
{
private:
    // A pointer to the data contained in the tensor
    void *data_;

public:
    TestNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims, TensorType tensor_type);

    ~TestNeuropodTensor();

    // Get a pointer to the underlying data
    TensorDataPointer get_data_ptr();
};

} // namespace neuropods
