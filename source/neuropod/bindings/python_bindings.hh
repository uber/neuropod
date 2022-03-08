/* Copyright (c) 2020 The Neuropod Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "neuropod/internal/neuropod_tensor.hh"
#include "neuropod/neuropod.hh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace neuropod
{

namespace py = pybind11;

// Convert a py:array to a NeuropodTensor
std::shared_ptr<NeuropodTensor> tensor_from_numpy(NeuropodTensorAllocator &allocator, py::array array);

// Convert a NeuropodTensor to a py::array
py::array tensor_to_numpy(std::shared_ptr<NeuropodTensor> value);

// Convert from a py::dict of numpy arrays to an unordered_map of `NeuropodTensor`s
NeuropodValueMap from_numpy_dict(NeuropodTensorAllocator &allocator, py::dict &items);

// Convert the items to a python dict of numpy arrays
py::dict to_numpy_dict(NeuropodValueMap &items);

} // namespace neuropod
