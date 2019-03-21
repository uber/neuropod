//
// Uber, Inc. (c) 2018
//

#include "test_neuropod_backend.hh"

#include "neuropods/internal/tensor_store.hh"

namespace neuropods
{

TestNeuropodBackend::TestNeuropodBackend() {}
TestNeuropodBackend::TestNeuropodBackend(const std::string &neuropod_path, std::unique_ptr<ModelConfig> &model_config) {}
TestNeuropodBackend::~TestNeuropodBackend() = default;

// Run inference
std::unique_ptr<TensorStore> TestNeuropodBackend::infer(const std::unordered_set<std::shared_ptr<NeuropodTensor>> &inputs)
{
    return stdx::make_unique<TensorStore>();
}

REGISTER_NEUROPOD_BACKEND(TestNeuropodBackend, "noop")

} // namespace neuropods
