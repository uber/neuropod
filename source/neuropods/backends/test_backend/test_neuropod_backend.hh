//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <vector>

#include "neuropods/backends/neuropod_backend.hh"

namespace neuropods
{

// A NeuropodBackend used in tests
class TestNeuropodBackend : public NeuropodBackend
{
public:
    TestNeuropodBackend();
    TestNeuropodBackend(const std::string &neuropod_path, std::unique_ptr<ModelConfig> model_config);
    ~TestNeuropodBackend();

    // Allocate a tensor of a specific type
    std::unique_ptr<NeuropodTensor> allocate_tensor(const std::string &         node_name,
                                                    const std::vector<int64_t> &input_dims,
                                                    TensorType                  tensor_type);

    // Run inference
    std::unique_ptr<TensorStore> infer(const TensorStore &inputs);
};
} // namespace neuropods
