//
// Uber, Inc. (c) 2019
//

#include "neuropod/internal/neuropod_tensor_raw_data_access.hh"

#include "neuropod/internal/neuropod_tensor.hh"

namespace neuropod
{

namespace internal
{

void *NeuropodTensorRawDataAccess::get_untyped_data_ptr(NeuropodTensor &tensor)
{
    return tensor.get_untyped_data_ptr();
}

const void *NeuropodTensorRawDataAccess::get_untyped_data_ptr(const NeuropodTensor &tensor)
{
    return tensor.get_untyped_data_ptr();
}

size_t NeuropodTensorRawDataAccess::get_bytes_per_element(const NeuropodTensor &tensor)
{
    return tensor.get_bytes_per_element();
}

} // namespace internal
} // namespace neuropod
