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
std::unique_ptr<NeuropodValueMap> TestNeuropodBackend::infer_internal(const SealedValueMap &inputs)
{
    return stdx::make_unique<NeuropodValueMap>();
}

struct SealedTestValueMap : public SealedValueMap
{
    void seal(const std::string &name, const std::shared_ptr<NeuropodValue> &item) {}
};

std::unique_ptr<SealedValueMap> TestNeuropodBackend::get_sealed_map()
{
    return stdx::make_unique<SealedTestValueMap>();
}

REGISTER_NEUROPOD_BACKEND(TestNeuropodBackend, "noop")

} // namespace neuropod
