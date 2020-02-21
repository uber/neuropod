//
// Uber, Inc. (c) 2018
//

#include "test_neuropod_backend.hh"

namespace neuropod
{

TestNeuropodBackend::TestNeuropodBackend()
{
    load_model();
}

TestNeuropodBackend::TestNeuropodBackend(const std::string &neuropod_path, const RuntimeOptions &options)
{
    load_model();
}

TestNeuropodBackend::~TestNeuropodBackend() = default;

void TestNeuropodBackend::load_model_internal() {}

// Run inference
std::unique_ptr<NeuropodValueMap> TestNeuropodBackend::infer_internal(const NeuropodValueMap &inputs)
{
    return stdx::make_unique<NeuropodValueMap>();
}

REGISTER_NEUROPOD_BACKEND(TestNeuropodBackend, "noop")

} // namespace neuropod
