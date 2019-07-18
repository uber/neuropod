//
// Uber, Inc. (c) 2019
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "neuropods/neuropods.hh"
#include "neuropods/backends/test_backend/test_neuropod_tensor.hh"
#include "neuropods/bindings/python_bindings.hh"
#include "neuropods/serialization/serialization.hh"

#include <sstream>

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

py::array deserialize_binding(py::bytes buffer)
{
    // Deserialize to a NeuropodTensor
    std::istringstream input_stream(buffer);
    auto allocator = DefaultTensorAllocator<neuropods::TestNeuropodTensor>();
    auto val = deserialize(input_stream, allocator);

    // Wrap it in a numpy array and return
    return tensor_to_numpy(std::dynamic_pointer_cast<NeuropodTensor>(val));
}

py::bytes serialize_binding(py::array numpy_array)
{
    // Wrap the numpy array in a NeuropodTensor
    auto allocator = DefaultTensorAllocator<neuropods::TestNeuropodTensor>();
    auto tensor = tensor_from_numpy(allocator, numpy_array);

    // Serialize the tensor
    std::stringstream buffer_stream;
    serialize(buffer_stream, *tensor);
    return py::bytes(buffer_stream.str());
}

} // namespace

PYBIND11_MODULE(neuropods_native, m) {
    py::class_<Neuropod>(m, "Neuropod")
        .def(py::init<const std::string &>())
        .def(py::init<const std::string &, const std::unordered_map<std::string, std::string> &>())
        .def(py::init<const std::string &, const std::string &>())
        .def("infer", &infer);

    m.def("serialize", &serialize_binding, "Convert a numpy array to a NeuropodTensor and serialize it");
    m.def("deserialize", &deserialize_binding, "Deserialize a string of bytes to a NeuropodTensor (and return it as a numpy array)");
}

} // namespace neuropods
