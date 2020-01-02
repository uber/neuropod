//
// Uber, Inc. (c) 2019
//

#include "neuropods/backends/test_backend/test_neuropod_tensor.hh"
#include "neuropods/bindings/python_bindings.hh"
#include "neuropods/neuropods.hh"
#include "neuropods/serialization/serialization.hh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace neuropod
{

namespace
{

py::dict infer(Neuropod &neuropod, py::dict &inputs_dict)
{
    // Convert from a py::dict of numpy arrays to an unordered_map of `NeuropodTensor`s
    auto             allocator = neuropod.get_tensor_allocator();
    NeuropodValueMap inputs    = from_numpy_dict(*allocator, inputs_dict);

    // Run inference
    auto outputs = neuropod.infer(inputs);

    // Convert the outputs to a python dict of numpy arrays
    return to_numpy_dict(*outputs);
}

py::array deserialize_tensor_binding(py::bytes buffer)
{
    // Deserialize to a NeuropodTensor
    std::istringstream input_stream(buffer);
    auto               allocator = DefaultTensorAllocator<TestNeuropodTensor>();
    auto               val       = deserialize<std::shared_ptr<NeuropodValue>>(input_stream, allocator);

    // Wrap it in a numpy array and return
    return tensor_to_numpy(std::dynamic_pointer_cast<NeuropodTensor>(val));
}

py::bytes serialize_tensor_binding(py::array numpy_array)
{
    // Wrap the numpy array in a NeuropodTensor
    auto allocator = DefaultTensorAllocator<neuropod::TestNeuropodTensor>();
    auto tensor    = tensor_from_numpy(allocator, numpy_array);

    // Serialize the tensor
    std::stringstream buffer_stream;
    serialize(buffer_stream, *tensor);
    return py::bytes(buffer_stream.str());
}

py::dict deserialize_valuemap_binding(py::bytes buffer)
{
    // Deserialize to a NeuropodTensor
    std::istringstream input_stream(buffer);
    auto               allocator = DefaultTensorAllocator<TestNeuropodTensor>();
    auto               val       = deserialize<NeuropodValueMap>(input_stream, allocator);

    // Wrap it in a numpy array and return
    return to_numpy_dict(val);
}

py::bytes serialize_valuemap_binding(py::dict items)
{
    // Wrap the numpy array in a NeuropodTensor
    auto allocator = DefaultTensorAllocator<neuropod::TestNeuropodTensor>();
    auto valuemap  = from_numpy_dict(allocator, items);

    // Serialize the tensor
    std::stringstream buffer_stream;
    serialize(buffer_stream, valuemap);
    return py::bytes(buffer_stream.str());
}

template <typename... Params>
std::unique_ptr<Neuropod> make_neuropod(py::kwargs kwargs, Params &&... params)
{
    RuntimeOptions options;

    if (kwargs.contains("visible_gpu"))
    {
        if (kwargs["visible_gpu"].is_none())
        {
            options.visible_device = Device::CPU;
        }
        else
        {
            options.visible_device = kwargs["visible_gpu"].cast<int>();
        }
    }

    return stdx::make_unique<Neuropod>(std::forward<Params>(params)..., options);
}

} // namespace

PYBIND11_MODULE(neuropod_native, m)
{
    py::class_<Neuropod>(m, "Neuropod")
        .def(py::init([](const std::string &path, py::kwargs kwargs) { return make_neuropod(kwargs, path); }))
        .def(py::init([](const std::string &                                 path,
                         const std::unordered_map<std::string, std::string> &default_backend_overrides,
                         py::kwargs kwargs) { return make_neuropod(kwargs, path, default_backend_overrides); }))
        .def(py::init([](const std::string &path, const std::string &backend_name, py::kwargs kwargs) {
            return make_neuropod(kwargs, path, backend_name);
        }))
        .def("infer", &infer);

    m.def("serialize", &serialize_tensor_binding, "Convert a numpy array to a NeuropodTensor and serialize it");
    m.def("deserialize",
          &deserialize_tensor_binding,
          "Deserialize a string of bytes to a NeuropodTensor (and return it as a numpy array)");

    m.def("serialize",
          &serialize_valuemap_binding,
          "Convert a dict of numpy arrays to a NeuropodValueMap and serialize it");
    m.def("deserialize_dict",
          &deserialize_valuemap_binding,
          "Deserialize a string of bytes to a NeuropodValueMap (and return it as a dict of numpy arrays)");
}

} // namespace neuropod
