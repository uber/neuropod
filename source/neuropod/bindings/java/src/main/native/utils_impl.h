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

namespace neuropod
{
namespace jni
{

template <typename T>
jobject createDirectBuffer(JNIEnv *env, NeuropodTensor *tensor)
{
    auto         typedTensor = tensor->as_typed_tensor<T>();
    if (typedTensor == nullptr)
    {
        throw std::runtime_error("the tensor does not match its type");
    }
    auto         rawPtr      = typedTensor->get_raw_data_ptr();
    const size_t memSize     = sizeof(T) * tensor->get_num_elements();
    return env->NewDirectByteBuffer(rawPtr, static_cast<jlong>(memSize));
}

} // namespace jni
} // namespace neuropod
