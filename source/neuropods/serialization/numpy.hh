//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/backends/tensor_allocator.hh"
#include "neuropods/backends/python_bridge/python_bridge.hh"
#include "neuropods/backends/python_bridge/numpy_neuropod_tensor.hh"

#include <boost/python.hpp>

namespace neuropods
{

namespace
{

struct allocate_from_tensor_visitor : public NeuropodTensorVisitor<std::shared_ptr<neuropods::NeuropodTensor>>
{
    template <typename T>
    std::shared_ptr<neuropods::NeuropodTensor> 
    operator()(const TypedNeuropodTensor<T> * tensor, neuropods::NeuropodTensorAllocator &allocator) const
    {
        std::shared_ptr<neuropods::TypedNeuropodTensor<T>> new_tensor = allocator.allocate_tensor<T>(tensor->get_dims());
        new_tensor->copy_from(tensor->get_data_as_vector());
        return new_tensor;
    }

    std::shared_ptr<neuropods::NeuropodTensor> 
    operator()(const TypedNeuropodTensor<std::string> * tensor, neuropods::NeuropodTensorAllocator &allocator) const
    {
        // std::unique_ptr<std::string> new_tensor;
        // new_tensor.reset(allocator.allocate_tensor<T>(tensor->get_type(), tensor->get_dims()));
        // new_tensor->as_typed_tensor<T>()->copy_from(tensor->get_data_as_vector());
        NEUROPOD_ERROR("Not implemented yet");
        return std::unique_ptr<neuropods::NeuropodTensor>();
    }

};

}

void save_to_npy(const std::string& filename, const NeuropodTensor &tensor)
{
    initialize_python_bridge_backend();

    neuropods::DefaultTensorAllocator<neuropods::NumpyNeuropodTensor> allocator;
    std::shared_ptr<neuropods::NeuropodTensor> new_tensor = tensor.apply_visitor(allocate_from_tensor_visitor{}, allocator);

    const auto* ndarray_holder = dynamic_cast<const NativeDataContainer<py::object>*>(new_tensor.get());
    assert(ndarray_holder);
    py::object ndarray = ndarray_holder->get_native_data();
    py::dict locals;
    locals["ndarray"] = ndarray;

    auto main_module    = py::import("__main__");
    auto main_namespace = main_module.attr("__dict__");

    std::stringstream npy_save_script;
    npy_save_script <<
        "import numpy as np\n"
        "np.save('" << filename << "', ndarray)\n";

    // Load the neuropod
    py::exec(npy_save_script.str().c_str(), main_namespace, locals);
}

}  // neuropods