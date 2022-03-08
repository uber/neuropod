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

#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/neuropod_tensor.hh"
#include "neuropod/internal/neuropod_tensor_raw_data_access.hh"
#include "neuropod/neuropod.hh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace neuropod
{

namespace py = pybind11;

namespace
{

TensorType get_array_type(py::array &array)
{
#define IS_INSTANCE_CHECK(cpp_type, neuropod_type)    \
    if (py::isinstance<py::array_t<cpp_type>>(array)) \
        return neuropod_type;

    FOR_EACH_TYPE_MAPPING_EXCEPT_STRING(IS_INSTANCE_CHECK)

    // Strings need to be handled separately because `py::isinstance` does not do
    // what we want in this case.
    if (array.dtype().kind() == 'S' || array.dtype().kind() == 'U')
    {
        return STRING_TENSOR;
    }

    NEUROPOD_ERROR("Unsupported array type in python bindings: {}", array.dtype().kind());
#undef IS_INSTANCE_CHECK
}

pybind11::dtype get_py_type(const NeuropodTensor &tensor)
{
#define GET_TYPE(CPP_TYPE, NEUROPOD_TYPE)       \
    case NEUROPOD_TYPE: {                       \
        return pybind11::dtype::of<CPP_TYPE>(); \
    }

    const auto &tensor_type = tensor.get_tensor_type();
    switch (tensor_type)
    {
        FOR_EACH_TYPE_MAPPING_EXCEPT_STRING(GET_TYPE)
    default:
        NEUROPOD_ERROR("Unsupported array type in python bindings: {}", tensor_type);
    }
#undef GET_TYPE
}

std::shared_ptr<NeuropodTensor> tensor_from_string_numpy(NeuropodTensorAllocator &allocator,
                                                         py::array &              array,
                                                         std::vector<int64_t> &   shape)
{
    // Unfortunately, for strings, we need to copy all the data in the tensor
    auto tensor  = allocator.allocate_tensor<std::string>(shape);
    int  max_len = array.itemsize();
    int  numel   = tensor->get_num_elements();

    // Get a pointer to the underlying data
    char *data = static_cast<char *>(array.mutable_data());

    std::vector<std::string> out;
    std::string              chars_to_strip("\0", 1);
    for (int i = 0; i < numel * max_len; i += max_len)
    {
        std::string item(data + i, max_len);

        // Remove null padding at the end
        item.erase(item.find_last_not_of(chars_to_strip) + 1);
        out.emplace_back(item);
    }

    // This potentially does another copy (depending on the backend)
    tensor->copy_from(out);

    return tensor;
}

} // namespace

std::shared_ptr<NeuropodTensor> tensor_from_numpy(NeuropodTensorAllocator &allocator, py::array array)
{
    // Make sure the array is contiguous and aligned
    // NOLINTNEXTLINE(readability-implicit-bool-conversion, hicpp-signed-bitwise)
    if (!(array.flags() & py::detail::npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_) ||
        !(array.flags() & py::detail::npy_api::constants::NPY_ARRAY_ALIGNED_))
    {
        SPDLOG_WARN("Expected numpy array to be contiguous and aligned; converting...");
        array = py::array::ensure(array,
                                  py::detail::npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_ |
                                      py::detail::npy_api::constants::NPY_ARRAY_ALIGNED_);
    }

    auto ndims = array.ndim();
    auto dims  = array.shape();
    auto dtype = get_array_type(array);
    auto data  = array.mutable_data();

    // Capture the array in our deleter so it doesn't get deallocated
    // until we're done
    auto to_delete = std::make_shared<py::array>(array);
    auto deleter   = [to_delete](void *unused) mutable {
        py::gil_scoped_acquire gil;
        to_delete.reset();
    };

    // Create a vector with the shape info
    std::vector<int64_t> shape(&dims[0], &dims[ndims]);

    // Handle string tensors
    if (dtype == STRING_TENSOR)
    {
        return tensor_from_string_numpy(allocator, array, shape);
    }

    // Wrap the data from the numpy array
    return allocator.tensor_from_memory(shape, dtype, data, deleter);
}

py::array tensor_to_numpy(std::shared_ptr<NeuropodTensor> value)
{
    auto tensor = value->as_tensor();

    // This isn't going to be null, but we do a null check to keep
    // static analyzers happy
    if (tensor == nullptr)
    {
        NEUROPOD_ERROR("Error converting value to tensor");
    }

    auto dims = tensor->get_dims();

    // Handle string tensors
    if (tensor->get_tensor_type() == STRING_TENSOR)
    {
        // Special case for empty string tensors because the pybind functions below don't correctly set the
        // type of the resulting array in this case
        if (tensor->get_num_elements() == 0)
        {
            return py::array_t<std::array<char, 1>>(dims);
        }

        // We need to return as bytes to python since we don't know what encoding this has

        // Get the data as a string vector
        auto data_vec = tensor->as_typed_tensor<std::string>()->get_data_as_vector();

        // Maybe there's a better way of doing this
        size_t max_item_size_bytes = 0;
        for (const auto &item : data_vec)
        {
            max_item_size_bytes = std::max(item.size(), max_item_size_bytes);
        }

        // Set up an array with the right format
        py::dtype dt(fmt::format("|S{}", max_item_size_bytes));
        auto      arr = py::array(dt, size_t{tensor->get_num_elements()});

        // Zero the whole array
        memset(arr.mutable_data(), 0, arr.nbytes());

        // Copy in the data
        // This is unfortunate, but because there isn't really a standard underlying string tensor format
        // across frameworks, we need to make a copy
        for (size_t i = 0; i < data_vec.size(); i++)
        {
            memcpy(arr.mutable_data(i), data_vec.at(i).data(), data_vec.at(i).size());
        }

        // Resize to target dims
        arr.resize(dims);
        return arr;
    }

    auto data = internal::NeuropodTensorRawDataAccess::get_untyped_data_ptr(*tensor);

    // Make sure we don't deallocate the tensor until the numpy array is deallocated
    auto deleter        = [value](void *unused) {};
    auto deleter_handle = register_deleter(deleter, nullptr);
    auto capsule        = py::capsule(deleter_handle, [](void *handle) { run_deleter(handle); });
    return py::array(get_py_type(*tensor), dims, data, capsule);
}

NeuropodValueMap from_numpy_dict(NeuropodTensorAllocator &allocator, py::dict &items)
{
    // Convert from a py::dict of numpy arrays to an unordered_map of `NeuropodTensor`s
    NeuropodValueMap out;
    for (auto item : items)
    {
        out[item.first.cast<std::string>()] = tensor_from_numpy(allocator, item.second.cast<py::array>());
    }

    return out;
}

py::dict to_numpy_dict(NeuropodValueMap &items)
{
    // Convert the items to a python dict of numpy arrays
    py::dict out;
    for (auto &item : items)
    {
        out[item.first.c_str()] = tensor_to_numpy(std::dynamic_pointer_cast<NeuropodTensor>(item.second));
    }

    return out;
}

} // namespace neuropod
