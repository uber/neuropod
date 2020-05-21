#include <torch/script.h>

at::Tensor add(at::Tensor x, at::Tensor y)
{
    return x + y;
}

#if CAFFE2_VERSION == 10100
static auto registry = torch::jit::RegisterOperators("neuropod_test_ops::add", &add);
#else
static auto registry = torch::RegisterOperators("neuropod_test_ops::add", &add);
#endif