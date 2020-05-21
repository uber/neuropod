from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# See https://pytorch.org/tutorials/advanced/cpp_extension.html#building-with-setuptools
# and https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html#building-with-setuptools

# We want to avoid depending on `libtorch_python` for torchscript custom ops
ext_module = CppExtension("addition_op", ["addition_op.cc"])
ext_module.libraries = [
    libname for libname in ext_module.libraries if libname != "torch_python"
]

setup(
    name="addition_op",
    ext_modules=[ext_module],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
