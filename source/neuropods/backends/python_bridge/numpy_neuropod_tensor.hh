//
// Uber, Inc. (c) 2018
//

#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <algorithm>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <string>
#include <vector>

#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/neuropod_tensor.hh"

namespace neuropods
{

namespace py = boost::python;

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

void deallocator(PyObject * capsule)
{
    // The tensor is being deallocated, run the deleter that the user provided
    auto handle = PyCapsule_GetPointer(capsule, nullptr);
    run_deleter(handle);
}

} // namespace

// Utility function to get an nparray from a boost object
PyArrayObject *get_nparray_from_obj(py::object boost_obj)
{
    setup_numpy_if_needed();

    PyObject *obj = boost_obj.ptr();
    if (obj == nullptr || !PyArray_Check(obj))
    {
        NEUROPOD_ERROR("Can't extract data from python object; expected numpy array");
    }

    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(obj);

    // Make sure it's contiguous and aligned
    // It is the caller's responsibility to decref the returned array
    Py_INCREF(PyArray_DESCR(arr));
    PyArrayObject *carr = reinterpret_cast<PyArrayObject *>(
        PyArray_FromAny(obj, PyArray_DESCR(arr), 0, 0, NPY_ARRAY_CARRAY_RO, nullptr));

    return carr;
}

// This class is internal to neuropods and should not be exposed
// to users
template <typename T>
class NumpyNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<py::object>
{
public:
    // Allocate a numpy array
    NumpyNeuropodTensor(const std::vector<int64_t> &dims) : TypedNeuropodTensor<T>(dims)
    {
        setup_numpy_if_needed();

        // Allocate the numpy array
        PyObject *obj = PyArray_SimpleNew(
            // num dims
            dims.size(),

            // The dimensions, on OSX npy_intp is 32 bit, so we need to reinterpret_cast
            reinterpret_cast<npy_intp *>(const_cast<int64_t *>(dims.data())),

            // numpy typenum
            get_numpy_type_from_neuropod_type(get_tensor_type_from_cpp<T>()));

        py::handle<>       handle(obj);
        py::numeric::array arr(handle);

        nparray_ = arr;
    };

    // Wrap existing memory
    NumpyNeuropodTensor(const std::vector<int64_t> &dims, void * data, const Deleter &deleter)
        : TypedNeuropodTensor<T>(dims)
    {
        setup_numpy_if_needed();

        // Allocate the numpy array
        PyObject *obj = PyArray_SimpleNewFromData(
            // num dims
            dims.size(),

            // The dimensions, on OSX npy_intp is 32 bit, so we need to reinterpret_cast
            reinterpret_cast<npy_intp *>(const_cast<int64_t *>(dims.data())),

            // numpy typenum
            get_numpy_type_from_neuropod_type(get_tensor_type_from_cpp<T>()),

            // Data
            data
        );

        // Make sure the deleter gets called when deallocating the numpy array
        auto deleter_handle = register_deleter(deleter, data);
        PyObject *capsule = PyCapsule_New(deleter_handle, nullptr, deallocator);
        PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(obj), capsule);

        py::handle<>       handle(obj);
        py::numeric::array arr(handle);

        nparray_ = arr;
    };

    // Wrap an existing array
    NumpyNeuropodTensor(PyArrayObject *nparray)
        : TypedNeuropodTensor<T>(get_dims_from_numpy(nparray))
    {
        py::handle<>       handle(reinterpret_cast<PyObject *>(nparray));
        py::numeric::array arr(handle);

        nparray_ = arr;
    }


    ~NumpyNeuropodTensor() = default;

    // Get a pointer to the underlying data
    T *get_raw_data_ptr()
    {
        auto arr = reinterpret_cast<PyArrayObject *>(nparray_.ptr());

        // Get a pointer to the underlying data
        void *data = PyArray_DATA(arr);

        return static_cast<T *>(data);
    }

    const T *get_raw_data_ptr() const
    {
        auto arr = reinterpret_cast<PyArrayObject *>(nparray_.ptr());

        // Get a pointer to the underlying data
        const void *data = PyArray_DATA(arr);

        return static_cast<const T *>(data);
    }

    py::object get_native_data() { return nparray_; }

    // The underlying numpy array
    py::object nparray_;
};


// Specialization for strings
template <>
class NumpyNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>, public NativeDataContainer<py::object>
{
public:
    // Allocate a numpy array
    NumpyNeuropodTensor(const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<std::string>(dims)
    {
        setup_numpy_if_needed();
    };

    // Wrap an existing array
    NumpyNeuropodTensor(PyArrayObject *nparray)
        : TypedNeuropodTensor<std::string>(get_dims_from_numpy(nparray))
    {
        py::handle<>       handle(reinterpret_cast<PyObject *>(nparray));
        py::numeric::array arr(handle);

        nparray_ = arr;
    }


    ~NumpyNeuropodTensor() = default;

    void set(const std::vector<std::string> &data)
    {
        // Get the length of the longest string
        size_t max_len = 0;
        for (const auto &item : data)
        {
            max_len = std::max(max_len, item.length());
        }

        // Allocate the numpy array
        const auto dims = get_dims();
        PyObject * obj  = PyArray_New(
            // Type
            &PyArray_Type,

            // num dims
            dims.size(),

            // The dimensions, on OSX npy_intp is 32 bit, so we need to reinterpret_cast
            reinterpret_cast<npy_intp *>(const_cast<int64_t *>(dims.data())),

            // numpy typenum
            NPY_STRING,

            // strides
            nullptr,

            // data
            nullptr,

            // itemsize
            max_len,

            // flags
            0,

            // PyObject
            nullptr);

        // Cast to a PyArrayObject
        auto pyarr = reinterpret_cast<PyArrayObject *>(obj);

        // Zero fill
        PyArray_FILLWBYTE(pyarr, 0);

        // Get a pointer to the underlying data
        char *arr_data = static_cast<char *>(PyArray_DATA(pyarr));

        // Set the data
        char *arr_data_ptr = arr_data;
        for (const auto &item : data)
        {
            memcpy(arr_data_ptr, item.c_str(), item.length() * sizeof(char));
            arr_data_ptr += max_len;
        }

        py::handle<>       handle(obj);
        py::numeric::array arr(handle);

        nparray_ = arr;
    }

    std::vector<std::string> get_data_as_vector()
    {
        auto arr = reinterpret_cast<PyArrayObject *>(nparray_.ptr());

        int max_len = PyArray_ITEMSIZE(arr);
        int numel   = get_num_elements();

        // Get a pointer to the underlying data
        char *data = static_cast<char *>(PyArray_DATA(arr));


        std::vector<std::string> out;
        std::string              chars_to_strip("\0", 1);
        for (int i = 0; i < numel * max_len; i += max_len)
        {
            std::string item(data + i, max_len);

            // Remove null padding at the end
            item.erase(item.find_last_not_of(chars_to_strip) + 1);
            out.emplace_back(item);
        }

        return out;
    }

    py::object get_native_data() { return nparray_; }

    // The underlying numpy array
    py::object nparray_;
};

} // namespace neuropods
