//
// Uber, Inc. (c) 2019
//

#pragma once

// Utilities used to access data in neuropod tensor objects as Eigen library Matrices and Vectors

#include "neuropods/internal/neuropod_tensor.hh"

#include <Eigen/Dense>

namespace neuropods
{

// Wraps neuropod tensor memory with Eigen matrix
template <typename _Scalar>
Eigen::Map<const Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_eigen(
    const neuropods::TypedNeuropodTensor<_Scalar> &tensor)
{
    const auto &dims = tensor.get_dims();

    if (dims.size() > 2)
    {
        NEUROPOD_ERROR("Only tensors with rank of 1 or 2 are supported by this function. "
                         "Tensor has rank of " << dims.size() << ".");
    }

    const auto rows = dims[0];
    const auto cols = dims.size() == 2 ? dims[1] : 1;

    return Eigen::Map<const Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        tensor.get_raw_data_ptr(), rows, cols);
}

// Wraps neuropod tensor memory with Eigen matrix
template <typename _Scalar>
Eigen::Map<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_eigen(
    neuropods::TypedNeuropodTensor<_Scalar> &tensor)
{
    const auto &dims = tensor.get_dims();

    if (dims.size() > 2)
    {
        NEUROPOD_ERROR("Only tensors with rank of 1 or 2 are supported by this function. "
                         "Tensor has rank of " << dims.size() << ".");
    }

    const auto rows = dims[0];
    const auto cols = dims.size() == 2 ? dims[1] : 1;

    return Eigen::Map<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        tensor.get_raw_data_ptr(), rows, cols);
}

// Wraps neuropod tensor memory with Eigen matrix
template <typename _Scalar>
Eigen::Map<const Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_eigen(
    const neuropods::NeuropodTensor &tensor)
{
    return as_eigen<_Scalar>(*tensor.as_typed_tensor<_Scalar>());
}

// Wraps neuropod tensor memory with Eigen matrix
template <typename _Scalar>
Eigen::Map<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_eigen(
    neuropods::NeuropodTensor &tensor)
{
    return as_eigen<_Scalar>(*tensor.as_typed_tensor<_Scalar>());
}

} // namespace neuropods
