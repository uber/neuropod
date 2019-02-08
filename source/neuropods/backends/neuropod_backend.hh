//
// Uber, Inc. (c) 2018
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "neuropods/internal/backend_registration.hh"
#include "neuropods/internal/tensor_types.hh"

namespace neuropods
{

class NeuropodTensor;
struct TensorStore;

// The interface that every neuropod backend implements
class NeuropodBackend
{
public:
    virtual ~NeuropodBackend() {}

    // Allocate a tensor of a specific type
    virtual std::unique_ptr<NeuropodTensor> allocate_tensor(const std::string &         node_name,
                                                            const std::vector<int64_t> &input_dims,
                                                            TensorType                  tensor_type)
        = 0;

    // Run inference
    virtual std::unique_ptr<TensorStore> infer(const TensorStore &inputs) = 0;
};
} // namespace neuropods
