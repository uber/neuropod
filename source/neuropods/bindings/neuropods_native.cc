//
// Uber, Inc. (c) 2019
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "neuropods/bindings/python_bindings.hh"
#include "neuropods/neuropods.hh"

namespace neuropods
{

namespace
{

py::dict infer(Neuropod &neuropod, py::dict &inputs_dict)
{
    // Convert from a py::dict of numpy arrays to an unordered_map of `NeuropodTensor`s
    auto allocator = neuropod.get_tensor_allocator();
    NeuropodValueMap inputs = from_numpy_dict(*allocator, inputs_dict);

    // Run inference
    auto outputs = neuropod.infer(inputs);

    // Convert the outputs to a python dict of numpy arrays
    return to_numpy_dict(*outputs);
}

} // namespace

PYBIND11_MODULE(neuropods_native, m) {
    py::class_<Neuropod>(m, "Neuropod")
        .def(py::init<const std::string &>())
        .def(py::init<const std::string &, const std::unordered_map<std::string, std::string> &>())
        .def(py::init<const std::string &, const std::string &>())
        .def("infer", &infer);

}

} // namespace neuropods
