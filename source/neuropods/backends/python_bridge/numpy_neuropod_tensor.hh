//
// Uber, Inc. (c) 2018
//

#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <string>
#include <vector>

#include "neuropods/internal/neuropod_tensor.hh"

namespace neuropods
{

namespace py = boost::python;

// Utility function to get an nparray from a boost object
PyArrayObject *get_nparray_from_obj(py::object boost_obj);

// This class is internal to neuropods and should not be exposed
// to users
class NumpyNeuropodTensor : public NeuropodTensor
{
public:
    // Allocate a numpy array
    NumpyNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims, TensorType tensor_type);

    // Wrap an existing array
    NumpyNeuropodTensor(const std::string &name, PyArrayObject *nparray);

    ~NumpyNeuropodTensor();

    // Get a pointer to the underlying data
    TensorDataPointer get_data_ptr();

    // The underlying numpy array
    py::object nparray;
};

} // namespace neuropods
