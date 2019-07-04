#include <torch/script.h>

at::Tensor add(at::Tensor x, at::Tensor y)
{
    return x + y;
}

static auto registry =
  torch::jit::RegisterOperators("neuropod_test_ops::add", &add);
