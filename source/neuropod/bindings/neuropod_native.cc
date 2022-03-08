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

#include "neuropod/bindings/python_bindings.hh"
#include "neuropod/core/generic_tensor.hh"
#include "neuropod/neuropod.hh"
#include "neuropod/serialization/serialization.hh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace neuropod
{

namespace
{

// A mapping between numpy types and Neuropod types
// TODO(vip): Share this with config_utils.cc
const std::unordered_map<std::string, TensorType> type_mapping = {
    {"float32", FLOAT_TENSOR},
    {"float64", DOUBLE_TENSOR},
    {"string", STRING_TENSOR},

    {"int8", INT8_TENSOR},
    {"int16", INT16_TENSOR},
    {"int32", INT32_TENSOR},
    {"int64", INT64_TENSOR},

    {"uint8", UINT8_TENSOR},
    {"uint16", UINT16_TENSOR},
    {"uint32", UINT32_TENSOR},
    {"uint64", UINT64_TENSOR},
};

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
    auto               allocator = get_generic_tensor_allocator();
    auto               val       = deserialize<std::shared_ptr<NeuropodValue>>(input_stream, *allocator);

    // Wrap it in a numpy array and return
    return tensor_to_numpy(std::dynamic_pointer_cast<NeuropodTensor>(val));
}

py::bytes serialize_tensor_binding(py::array numpy_array)
{
    // Wrap the numpy array in a NeuropodTensor
    auto allocator = get_generic_tensor_allocator();
    auto tensor    = tensor_from_numpy(*allocator, numpy_array);

    // Serialize the tensor
    std::stringstream buffer_stream;
    serialize(buffer_stream, *tensor);
    return py::bytes(buffer_stream.str());
}

py::dict deserialize_valuemap_binding(py::bytes buffer)
{
    // Deserialize to a NeuropodTensor
    std::istringstream input_stream(buffer);
    auto               allocator = get_generic_tensor_allocator();
    auto               val       = deserialize<NeuropodValueMap>(input_stream, *allocator);

    // Wrap it in a numpy array and return
    return to_numpy_dict(val);
}

py::bytes serialize_valuemap_binding(py::dict items)
{
    // Wrap the numpy array in a NeuropodTensor
    auto allocator = get_generic_tensor_allocator();
    auto valuemap  = from_numpy_dict(*allocator, items);

    // Serialize the tensor
    std::stringstream buffer_stream;
    serialize(buffer_stream, valuemap);
    return py::bytes(buffer_stream.str());
}

RuntimeOptions get_options_from_kwargs(py::kwargs &kwargs)
{
    RuntimeOptions options;

    for (const auto &item : kwargs)
    {
        const auto  key   = item.first.cast<std::string>();
        const auto &value = item.second;

        if (key == "visible_gpu")
        {
            if (value.is_none())
            {
                options.visible_device = Device::CPU;
            }
            else
            {
                options.visible_device = value.cast<int>();
            }
        }
        else if (key == "use_ope")
        {
            options.use_ope = value.cast<bool>();
        }
        else
        {
            NEUROPOD_ERROR("Got unexpected keyword argument {}", key);
        }
    }

    return options;
}

template <typename... Params>
std::unique_ptr<Neuropod> make_neuropod(py::kwargs kwargs, Params &&... params)
{
    auto options = get_options_from_kwargs(kwargs);
    return stdx::make_unique<Neuropod>(std::forward<Params>(params)..., options);
}

} // namespace

PYBIND11_MODULE(neuropod_native, m)
{
    py::class_<Neuropod>(m, "Neuropod")
        .def(py::init([](const std::string &path, py::kwargs kwargs) { return make_neuropod(kwargs, path); }))
        .def(py::init([](const std::string &                 path,
                         const std::vector<BackendLoadSpec> &default_backend_overrides,
                         py::kwargs kwargs) { return make_neuropod(kwargs, path, default_backend_overrides); }))
        .def("infer", &infer)
        .def("get_inputs", &Neuropod::get_inputs)
        .def("get_outputs", &Neuropod::get_outputs)
        .def("get_name", &Neuropod::get_name)
        .def("get_platform", &Neuropod::get_platform);

    py::class_<TensorSpec>(m, "TensorSpec")
        .def_readonly("name", &TensorSpec::name)
        .def_readonly("type", &TensorSpec::type)
        .def_readonly("dims", &TensorSpec::dims);

    py::class_<Dimension>(m, "Dimension")
        .def_readonly("value", &Dimension::value)
        .def_readonly("symbol", &Dimension::symbol);

    auto type_enum = py::enum_<TensorType>(m, "TensorType");
    for (const auto &item : type_mapping)
    {
        type_enum = type_enum.value(item.first.c_str(), item.second);
    }

    py::class_<BackendLoadSpec>(m, "BackendLoadSpec")
        .def(py::init<const std::string &, const std::string &, const std::string &>());

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
