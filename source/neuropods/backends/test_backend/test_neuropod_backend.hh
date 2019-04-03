//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <vector>

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/backends/test_backend/test_neuropod_tensor.hh"

namespace neuropods
{

// A NeuropodBackend used in tests
class TestNeuropodBackend : public NeuropodBackendWithDefaultAllocator<TestNeuropodTensor>
{
public:
    TestNeuropodBackend();
    TestNeuropodBackend(const std::string &neuropod_path, std::unique_ptr<ModelConfig> &model_config);
    ~TestNeuropodBackend();

    // Run inference
    std::unique_ptr<TensorMap> infer(const std::unordered_set<std::shared_ptr<NeuropodTensor>> &inputs);
};
} // namespace neuropods
