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

#include "neuropod/backends/tensor_allocator.hh"
#include "neuropod/internal/neuropod_tensor.hh"
#include "neuropod/internal/neuropod_tensor_raw_data_access.hh"
#include "neuropod/serialization/serialization.hh"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>

namespace neuropod
{

namespace
{

void serialize_tensor(const NeuropodTensor &tensor, boost::archive::binary_oarchive &ar)
{
    int tensor_type = static_cast<int>(tensor.get_tensor_type());
    ar << tensor_type;
    ar << tensor.get_dims();

    if (tensor.get_tensor_type() == STRING_TENSOR)
    {
        const auto data = tensor.as_typed_tensor<std::string>()->get_data_as_vector();
        ar << data;
    }
    else
    {
        ar << boost::serialization::make_array(
            static_cast<const uint8_t *>(internal::NeuropodTensorRawDataAccess::get_untyped_data_ptr(tensor)),
            tensor.get_num_elements() * internal::NeuropodTensorRawDataAccess::get_bytes_per_element(tensor));
    }
}

std::shared_ptr<NeuropodTensor> deserialize_tensor(boost::archive::binary_iarchive &ar,
                                                   NeuropodTensorAllocator &        allocator)
{
    int                  type;
    std::vector<int64_t> dims;
    ar >> type;
    ar >> dims;

    auto out = allocator.allocate_tensor(dims, static_cast<TensorType>(type));
    if (out->get_tensor_type() == STRING_TENSOR)
    {
        std::vector<std::string> data;
        ar >> data;
        out->as_typed_tensor<std::string>()->copy_from(data);
    }
    else
    {
        ar >> boost::serialization::make_array(
                  static_cast<uint8_t *>(internal::NeuropodTensorRawDataAccess::get_untyped_data_ptr(*out)),
                  out->get_num_elements() * internal::NeuropodTensorRawDataAccess::get_bytes_per_element(*out));
    }

    return out;
}

MAKE_SERIALIZABLE(NeuropodTensor, serialize_tensor, deserialize_tensor);

} // namespace

} // namespace neuropod
