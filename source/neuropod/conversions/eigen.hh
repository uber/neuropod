/* Copyright (c) 2020 UATC, LLC

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

#pragma once

// Utilities used to access data in neuropod tensor objects as Eigen library Matrices and Vectors

#include "neuropod/internal/error_utils_header_only.hh"
#include "neuropod/internal/neuropod_tensor.hh"

#include <Eigen/Dense>

namespace neuropod
{

// Wraps neuropod tensor memory with Eigen matrix
template <typename _Scalar>
Eigen::Map<const Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_eigen(
    const neuropod::TypedNeuropodTensor<_Scalar> &tensor)
{
    const auto &dims = tensor.get_dims();

    if (dims.size() > 2)
    {
        NEUROPOD_ERROR_HH("Only tensors with rank of 1 or 2 are supported by this function. "
                          "Tensor has rank of {}.",
                          dims.size());
    }

    const auto rows = dims[0];
    const auto cols = dims.size() == 2 ? dims[1] : 1;

    return Eigen::Map<const Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        tensor.get_raw_data_ptr(), rows, cols);
}

// Wraps neuropod tensor memory with Eigen matrix
template <typename _Scalar>
Eigen::Map<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_eigen(
    neuropod::TypedNeuropodTensor<_Scalar> &tensor)
{
    const auto &dims = tensor.get_dims();

    if (dims.size() > 2)
    {
        NEUROPOD_ERROR_HH("Only tensors with rank of 1 or 2 are supported by this function. "
                          "Tensor has rank of {}.",
                          dims.size());
    }

    const auto rows = dims[0];
    const auto cols = dims.size() == 2 ? dims[1] : 1;

    return Eigen::Map<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        tensor.get_raw_data_ptr(), rows, cols);
}

// Wraps neuropod tensor memory with Eigen matrix
template <typename _Scalar>
Eigen::Map<const Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_eigen(
    const neuropod::NeuropodTensor &tensor)
{
    return as_eigen<_Scalar>(*tensor.as_typed_tensor<_Scalar>());
}

// Wraps neuropod tensor memory with Eigen matrix
template <typename _Scalar>
Eigen::Map<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> as_eigen(
    neuropod::NeuropodTensor &tensor)
{
    return as_eigen<_Scalar>(*tensor.as_typed_tensor<_Scalar>());
}

} // namespace neuropod
