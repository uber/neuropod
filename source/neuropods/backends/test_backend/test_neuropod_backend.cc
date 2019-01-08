//
// Uber, Inc. (c) 2018
//

#include "test_neuropod_backend.hh"

#include "neuropods/backends/test_backend/test_neuropod_tensor.hh"
#include "neuropods/internal/tensor_store.hh"

namespace neuropods
{

TestNeuropodBackend::TestNeuropodBackend() {}
TestNeuropodBackend::TestNeuropodBackend(const std::string &neuropod_path, std::unique_ptr<ModelConfig> model_config) {}
TestNeuropodBackend::~TestNeuropodBackend() = default;

// Allocate a tensor of a specific type
std::unique_ptr<NeuropodTensor> TestNeuropodBackend::allocate_tensor(const std::string &         node_name,
                                                                     const std::vector<int64_t> &input_dims,
                                                                     TensorType                  tensor_type)
{
    return make_tensor<TestNeuropodTensor>(tensor_type, node_name, input_dims);
}

// Run inference
std::unique_ptr<TensorStore> TestNeuropodBackend::infer(const TensorStore &inputs)
{
    return std::make_unique<TensorStore>();
}

REGISTER_NEUROPOD_BACKEND(TestNeuropodBackend, "noop")

} // namespace neuropods
