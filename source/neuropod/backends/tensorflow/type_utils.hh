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

#include "neuropod/internal/tensor_types.hh"
#include "tensorflow/core/framework/types.pb.h"

namespace neuropod
{

TensorType           get_neuropod_type_from_tf_type(tensorflow::DataType type);
tensorflow::DataType get_tf_type_from_neuropod_type(TensorType type);

} // namespace neuropod
