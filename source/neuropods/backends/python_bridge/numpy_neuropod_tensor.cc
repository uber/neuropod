//
// Uber, Inc. (c) 2018
//

#include "numpy_neuropod_tensor.hh"

#include <sstream>
#include <stdexcept>

#include "neuropods/backends/python_bridge/type_utils.hh"

namespace neuropods
{

namespace
{

void setup_numpy_if_needed()
{
    // Needed in order for numpy arrays to work from c++
    if (PyArray_API == nullptr)
    {
        import_array();
        py::numeric::array::set_module_and_type("numpy", "ndarray");
    }
}

std::vector<int64_t> get_dims_from_numpy(PyArrayObject *nparray)
{
    // Get info about the numpy array
    int       ndims = PyArray_NDIM(nparray);
    npy_intp *dims  = PyArray_DIMS(nparray);

    // Create a vector with the shape info
    std::vector<int64_t> shape(&dims[0], &dims[ndims]);

    return shape;
}

} // namespace

// Utility function to get an nparray from a boost object
PyArrayObject *get_nparray_from_obj(py::object boost_obj)
{
    setup_numpy_if_needed();

    PyObject *obj = boost_obj.ptr();
    if (obj == nullptr || !PyArray_Check(obj))
    {
        throw std::runtime_error("Can't extract data from python object; expected numpy array");
    }

    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(obj);

    // Make sure it's contiguous and aligned
    // It is the caller's responsibility to decref the returned array
    Py_INCREF(PyArray_DESCR(arr));
    PyArrayObject *carr = reinterpret_cast<PyArrayObject *>(
        PyArray_FromAny(obj, PyArray_DESCR(arr), 0, 0, NPY_ARRAY_CARRAY_RO, nullptr));

    return carr;
}

// Allocate a new numpy array
template <typename T>
NumpyNeuropodTensor<T>::NumpyNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
    : TypedNeuropodTensor<T>(name, dims)
{
    setup_numpy_if_needed();

    // Allocate the numpy array
    PyObject *obj = PyArray_SimpleNew(
        // num dims
        dims.size(),

        // The dimensions
        const_cast<npy_intp *>(dims.data()),

        // numpy typenum
        get_numpy_type_from_neuropod_type(get_tensor_type_from_cpp<T>()));

    py::handle<>       handle(obj);
    py::numeric::array arr(handle);

    nparray = arr;
};

// Wrap an existing array
template <typename T>
NumpyNeuropodTensor<T>::NumpyNeuropodTensor(const std::string &name, PyArrayObject *obj)
    : TypedNeuropodTensor<T>(name, get_dims_from_numpy(obj))
{
    py::handle<>       handle(reinterpret_cast<PyObject *>(obj));
    py::numeric::array arr(handle);

    nparray = arr;
}

template <typename T>
NumpyNeuropodTensor<T>::~NumpyNeuropodTensor() = default;

template <typename T>
T *NumpyNeuropodTensor<T>::get_raw_data_ptr()
{
    auto arr = reinterpret_cast<PyArrayObject *>(nparray.ptr());

    // Get a pointer to the underlying data
    void *data = PyArray_DATA(arr);

    return static_cast<T *>(data);
}

template <typename T>
py::object NumpyNeuropodTensor<T>::get_native_data()
{
    return nparray;
}

// Instantiate the templates
#define INIT_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE) template class NumpyNeuropodTensor<CPP_TYPE>;

FOR_EACH_TYPE_MAPPING(INIT_TEMPLATES_FOR_TYPE);

} // namespace neuropods
