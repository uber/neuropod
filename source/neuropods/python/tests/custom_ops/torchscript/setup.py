from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# See https://pytorch.org/tutorials/advanced/cpp_extension.html#building-with-setuptools
# and https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html#building-with-setuptools
setup(
    name="addition_op",
    ext_modules=[CppExtension("addition_op", ["addition_op.cc"]),],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
