//
// Uber, Inc. (c) 2018
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "neuropod_tensor.hh"

namespace neuropods
{

// A structure to store tensors
// This is opaque to users of Neuropods
struct TensorStore
{
    std::vector<std::shared_ptr<NeuropodTensor>> tensors;

    // Get a tensor by name
    std::shared_ptr<NeuropodTensor> find(const std::string &name);
};

} // namespace neuropods
