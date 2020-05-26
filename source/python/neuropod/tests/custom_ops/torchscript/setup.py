# Copyright (c) 2020 UATC, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
