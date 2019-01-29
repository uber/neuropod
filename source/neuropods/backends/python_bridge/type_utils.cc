//
// Uber, Inc. (c) 2018
//

#include "type_utils.hh"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <sstream>
#include <stdexcept>

namespace neuropods
{

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

    std::stringstream ss;
    ss << "Unsupported Neuropod type: " << type;
    throw std::runtime_error(ss.str());
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

} // namespace neuropods
