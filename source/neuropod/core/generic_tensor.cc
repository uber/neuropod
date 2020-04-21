//
// Uber, Inc. (c) 2020
//

#include "neuropod/core/generic_tensor.hh"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace neuropod
{

std::unique_ptr<NeuropodTensorAllocator> get_generic_tensor_allocator()
{
    return stdx::make_unique<DefaultTensorAllocator<GenericNeuropodTensor>>();
}

} // namespace neuropod
