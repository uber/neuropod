//
// Uber, Inc. (c) 2018
//

#include "test_neuropod_backend.hh"

namespace neuropod
{

TestNeuropodBackend::TestNeuropodBackend() = default;
TestNeuropodBackend::TestNeuropodBackend(const std::string &neuropod_path, const RuntimeOptions &options) {}
TestNeuropodBackend::~TestNeuropodBackend() = default;

// Run inference
std::unique_ptr<NeuropodValueMap> TestNeuropodBackend::infer_internal(const NeuropodValueMap &inputs)
{
    return stdx::make_unique<NeuropodValueMap>();
}

REGISTER_NEUROPOD_BACKEND(TestNeuropodBackend, "noop")

} // namespace neuropod
