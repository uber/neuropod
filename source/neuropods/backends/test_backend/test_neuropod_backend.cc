//
// Uber, Inc. (c) 2018
//

#include "test_neuropod_backend.hh"

namespace neuropods
{

TestNeuropodBackend::TestNeuropodBackend() {}
TestNeuropodBackend::TestNeuropodBackend(const std::string &neuropod_path, std::unique_ptr<ModelConfig> &model_config) {}
TestNeuropodBackend::~TestNeuropodBackend() = default;

// Run inference
std::unique_ptr<TensorMap> TestNeuropodBackend::infer(const TensorSet &inputs)
{
    return stdx::make_unique<TensorMap>();
}

REGISTER_NEUROPOD_BACKEND(TestNeuropodBackend, "noop")

} // namespace neuropods
