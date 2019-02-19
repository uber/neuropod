//
// Uber, Inc. (c) 2018
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "neuropods/internal/neuropod_tensor.hh"

namespace neuropods
{

// A structure to store tensors
struct TensorStore
{
    std::vector<std::shared_ptr<NeuropodTensor>> tensors;

    // Get a tensor by name. Returns a nullptr if not found
    std::shared_ptr<NeuropodTensor> find(const std::string &name);

    // Get a tensor by name. Throws an error if not found
    std::shared_ptr<NeuropodTensor> find_or_throw(const std::string &name);
};

} // namespace neuropods
