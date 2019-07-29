#include <torch/extension.h>

at::Tensor add_forward(at::Tensor x, at::Tensor y)
{
    return x + y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &add_forward, "Add forward");
}
