//
// Uber, Inc. (c) 2019
//

#pragma once

#include <cstdlib>

namespace neuropod
{

class NeuropodTensor;

namespace internal
{

// This struct is used internally within the library to access raw untyped data
// from a NeuropodTensor
//
// This should NOT be used externally
struct NeuropodTensorRawDataAccess
{
    static void *get_untyped_data_ptr(NeuropodTensor &tensor);

    static const void *get_untyped_data_ptr(const NeuropodTensor &tensor);

    static size_t get_bytes_per_element(const NeuropodTensor &tensor);
};

} // namespace internal
} // namespace neuropod
