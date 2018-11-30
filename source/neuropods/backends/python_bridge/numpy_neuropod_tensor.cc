//
// Uber, Inc. (c) 2018
//

#include "numpy_neuropod_tensor.hh"

#include <sstream>
#include <stdexcept>

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

#define FOR_NP_NEUROPOD_MAPPING(FN) \
    FN(FLOAT_TENSOR, NPY_FLOAT)     \
    FN(DOUBLE_TENSOR, NPY_DOUBLE)   \
    FN(STRING_TENSOR, NPY_STRING)   \
                                    \
    FN(INT8_TENSOR, NPY_INT8)       \
    FN(INT16_TENSOR, NPY_INT16)     \
    FN(INT32_TENSOR, NPY_INT32)     \
    FN(INT64_TENSOR, NPY_INT64)     \
                                    \
    FN(UINT8_TENSOR, NPY_UINT8)     \
    FN(UINT16_TENSOR, NPY_UINT16)   \
    FN(UINT32_TENSOR, NPY_UINT32)   \
    FN(UINT64_TENSOR, NPY_UINT64)


int get_numpy_type_from_neuropod_type(TensorType type)
{
#define NEUROPOD_TO_NUMPY(NEUROPOD_TYPE, NUMPY_TYPE) \
    case NEUROPOD_TYPE:                              \
        return NUMPY_TYPE;

    switch (type)
    {
        FOR_NP_NEUROPOD_MAPPING(NEUROPOD_TO_NUMPY)
    }
}


TensorType get_neuropod_type_from_numpy_type(int type)
{
#define NUMPY_TO_NEUROPOD(NEUROPOD_TYPE, NUMPY_TYPE) \
    case NUMPY_TYPE:                                 \
        return NEUROPOD_TYPE;

    switch (type)
    {
        FOR_NP_NEUROPOD_MAPPING(NUMPY_TO_NEUROPOD)
    }

    std::stringstream ss;
    ss << "Unsupported numpy type: " << type;
    throw std::runtime_error(ss.str());
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
NumpyNeuropodTensor::NumpyNeuropodTensor(const std::string &         name,
                                         const std::vector<int64_t> &dims,
                                         TensorType                  tensor_type)
    : NeuropodTensor(name, tensor_type, dims)
{
    setup_numpy_if_needed();

    // Allocate the numpy array
    PyObject *obj = PyArray_SimpleNew(
        // num dims
        dims.size(),

        // The dimensions
        const_cast<npy_intp *>(dims.data()),

        // numpy typenum
        get_numpy_type_from_neuropod_type(tensor_type));

    py::handle<>       handle(obj);
    py::numeric::array arr(handle);

    nparray = arr;
};

// Wrap an existing array
NumpyNeuropodTensor::NumpyNeuropodTensor(const std::string &name, PyArrayObject *obj)
    : NeuropodTensor(name, get_neuropod_type_from_numpy_type(PyArray_TYPE(obj)), get_dims_from_numpy(obj))
{
    py::handle<>       handle(reinterpret_cast<PyObject *>(obj));
    py::numeric::array arr(handle);

    nparray = arr;
}


NumpyNeuropodTensor::~NumpyNeuropodTensor() = default;

TensorDataPointer NumpyNeuropodTensor::get_data_ptr()
{
    auto arr = reinterpret_cast<PyArrayObject *>(nparray.ptr());

    // Get a pointer to the underlying data
    void *data = PyArray_DATA(arr);

#define CAST_TENSOR(CPP_TYPE, NEUROPOD_TYPE)  \
    case NEUROPOD_TYPE:                       \
    {                                         \
        return static_cast<CPP_TYPE *>(data); \
    }

    // Cast it to the correct type and return
    switch (get_tensor_type())
    {
        FOR_EACH_TYPE_MAPPING(CAST_TENSOR)
    }
}

} // namespace neuropods
