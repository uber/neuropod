#include <torch/extension.h>

// This op is named the same thing as the other addition op
// but is not identical so it should cause a conflict
volatile int something;

at::Tensor add_forward(at::Tensor x, at::Tensor y)
{
    return x + y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &add_forward, "Add forward");
}
