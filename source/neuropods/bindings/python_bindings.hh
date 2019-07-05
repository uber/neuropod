//
// Uber, Inc. (c) 2019
//

#include <pybind11/pybind11.h>

#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/neuropods.hh"

namespace neuropods
{

namespace py = pybind11;

// Convert from a py::dict of numpy arrays to an unordered_map of `NeuropodTensor`s
NeuropodValueMap from_numpy_dict(NeuropodTensorAllocator &allocator, py::dict &items);

// Convert the items to a python dict of numpy arrays
py::dict to_numpy_dict(NeuropodValueMap &items);

} // namespace neuropods
