//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/backends/test_backend/test_neuropod_tensor.hh"

#include <string>
#include <vector>

namespace neuropod
{

// A NeuropodBackend used in tests
class TestNeuropodBackend : public NeuropodBackendWithDefaultAllocator<TestNeuropodTensor>
{
public:
    TestNeuropodBackend();
    TestNeuropodBackend(const std::string &neuropod_path, const RuntimeOptions &options);
    ~TestNeuropodBackend();

protected:
    // Run inference
    std::unique_ptr<NeuropodValueMap> infer_internal(const NeuropodValueMap &inputs);
};
} // namespace neuropod
