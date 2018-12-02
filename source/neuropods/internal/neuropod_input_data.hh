//
// Uber, Inc. (c) 2018
//

#pragma once

#include <memory>

#include "neuropods/fwd_declarations.hh"
#include "neuropods/internal/tensor_store.hh"

namespace neuropods
{

struct NeuropodInputData
{
    std::unique_ptr<TensorStore> data;
};

} // namespace neuropods