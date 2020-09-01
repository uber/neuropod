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

#include <stdexcept>

namespace neuropod
{
namespace jni
{

template <typename T>
jobject createDirectBuffer(JNIEnv *env, NeuropodTensor *tensor)
{
    auto typedTensor = tensor->as_typed_tensor<T>();
    if (typedTensor == nullptr)
    {
        throw std::runtime_error("tensor cannot be represented as requested type");
    }
    auto   rawPtr  = typedTensor->get_raw_data_ptr();
    size_t memSize = sizeof(T) * tensor->get_num_elements();
    return env->NewDirectByteBuffer(rawPtr, static_cast<jlong>(memSize));
}

template <size_t N>
void mapStringTensor(neuropod::TensorAccessor<neuropod::TypedNeuropodTensor<std::string> &, N> accessor,
                     const std::function<void(string_accessor_type *)> &                       func,
                     const std::vector<int64_t>                                                dims)
{
    for (int i = 0; i < dims[N - 1]; i++)
    {
        {
            mapStringTensor(accessor[i], func, dims);
        }
    }
}

template <>
inline void mapStringTensor(neuropod::TensorAccessor<neuropod::TypedNeuropodTensor<std::string> &, 1> accessor,
                            const std::function<void(string_accessor_type *)> &                       func,
                            const std::vector<int64_t>                                                dims)
{
    for (int i = 0; i < dims[0]; i++)
    {
        {
            auto elementAcc = accessor[i];
            func(&elementAcc);
        }
    }
}

template <size_t N>
void atStringTensor(neuropod::TensorAccessor<neuropod::TypedNeuropodTensor<std::string> &, N> accessor,
                    const std::function<void(string_accessor_type *)> &                       func,
                    const std::vector<int64_t>                                                targetDim)
{
    atStringTensor(accessor[targetDim[N - 1]], func, targetDim);
}

template <>
inline void atStringTensor(neuropod::TensorAccessor<neuropod::TypedNeuropodTensor<std::string> &, 1> accessor,
                           const std::function<void(string_accessor_type *)> &                       func,
                           const std::vector<int64_t>                                                targetDim)
{
    auto elementAcc = accessor[targetDim[0]];
    func(&elementAcc);
}

} // namespace jni
} // namespace neuropod
