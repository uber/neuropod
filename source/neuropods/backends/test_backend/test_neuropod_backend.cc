//
// Uber, Inc. (c) 2018
//

#include "test_neuropod_backend.hh"

namespace neuropods
{

TestNeuropodBackend::TestNeuropodBackend() {}
TestNeuropodBackend::TestNeuropodBackend(const std::string &           neuropod_path,
                                         std::unique_ptr<ModelConfig> &model_config,
                                         const RuntimeOptions &        options)
{
}
TestNeuropodBackend::~TestNeuropodBackend() = default;

// Run inference
std::unique_ptr<NeuropodValueMap> TestNeuropodBackend::infer(const NeuropodValueMap &inputs)
{
    return stdx::make_unique<NeuropodValueMap>();
}

REGISTER_NEUROPOD_BACKEND(TestNeuropodBackend, "noop")

} // namespace neuropods
