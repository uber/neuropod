//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/backends/test_backend/test_neuropod_tensor.hh"

#include <string>
#include <vector>

namespace neuropod
{

// A NeuropodBackend used in tests
class TestNeuropodBackend : public NeuropodBackendWithDefaultAllocator<TestNeuropodTensor>
{
public:
    TestNeuropodBackend();
    TestNeuropodBackend(const std::string &           neuropod_path,
                        std::unique_ptr<ModelConfig> &model_config,
                        const RuntimeOptions &        options);
    ~TestNeuropodBackend();

    // Run inference
    std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &inputs);
};
} // namespace neuropod
