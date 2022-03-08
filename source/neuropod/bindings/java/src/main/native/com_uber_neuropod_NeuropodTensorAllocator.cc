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

#include "com_uber_neuropod_NeuropodTensorAllocator.h"

#include "jclass_register.h"
#include "neuropod/neuropod.hh"
#include "utils.h"

#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#include <jni.h>

namespace njni = neuropod::jni;

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for Java API methods
JNIEXPORT void JNICALL Java_com_uber_neuropod_NeuropodTensorAllocator_nativeDelete(JNIEnv *env,
                                                                                   jobject /* unused */,
                                                                                   jlong handle)
{
    try
    {
        // Java NeuropodTensorAllocator stores a pointer to a shared_ptr
        auto allocatorPtr = reinterpret_cast<std::shared_ptr<neuropod::NeuropodTensorAllocator> *>(handle);
        std::unique_ptr<std::shared_ptr<neuropod::NeuropodTensorAllocator>> scopeHolder(allocatorPtr);
        allocatorPtr->reset();
    }
    catch (const std::exception &e)
    {
        njni::throw_java_exception(env, e.what());
    }
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for Java API methods
JNIEXPORT jlong JNICALL Java_com_uber_neuropod_NeuropodTensorAllocator_nativeAllocate(
    JNIEnv *env, jclass /* unused */, jlongArray dims, jint typeNumber, jobject buffer, jlong handle)
{
    try
    {
        auto allocator = *reinterpret_cast<std::shared_ptr<neuropod::NeuropodTensorAllocator> *>(handle);

        // Prepare shape
        jsize                size = env->GetArrayLength(dims);
        jlong *              arr  = env->GetLongArrayElements(dims, nullptr);
        std::vector<int64_t> shapes(arr, arr + size);
        env->ReleaseLongArrayElements(dims, arr, JNI_ABORT);
        // Prepare Buffer
        auto                    globalBufferRef = env->NewGlobalRef(buffer);
        auto                    bufferAddress   = env->GetDirectBufferAddress(buffer);
        const neuropod::Deleter deleter         = [globalBufferRef, env](void *unused) mutable {
            env->DeleteGlobalRef(globalBufferRef);
        };
        std::shared_ptr<neuropod::NeuropodValue> tensor;
        switch (static_cast<neuropod::TensorType>(typeNumber))
        {
        case neuropod::INT32_TENSOR: {
            tensor =
                allocator->tensor_from_memory<int32_t>(shapes, reinterpret_cast<int32_t *>(bufferAddress), deleter);
            break;
        }
        case neuropod::INT64_TENSOR: {
            tensor =
                allocator->tensor_from_memory<int64_t>(shapes, reinterpret_cast<int64_t *>(bufferAddress), deleter);
            break;
        }
        case neuropod::FLOAT_TENSOR: {
            tensor = allocator->tensor_from_memory<float>(shapes, reinterpret_cast<float *>(bufferAddress), deleter);
            break;
        }
        case neuropod::DOUBLE_TENSOR: {
            tensor = allocator->tensor_from_memory<double>(shapes, reinterpret_cast<double *>(bufferAddress), deleter);
            break;
        }
        default:
            throw std::runtime_error("unsupported tensor type");
        }
        return reinterpret_cast<jlong>(njni::toHeap(tensor));
    }
    catch (const std::exception &e)
    {
        njni::throw_java_exception(env, e.what());
    }
    return reinterpret_cast<jlong>(nullptr);
}

JNIEXPORT jlong JNICALL Java_com_uber_neuropod_NeuropodTensorAllocator_nativeCreateStringTensor(
    JNIEnv *env, jclass, jobject data, jlongArray dims, jlong allocatorHandle)
{
    try
    {
        auto allocator = *reinterpret_cast<std::shared_ptr<neuropod::NeuropodTensorAllocator> *>(allocatorHandle);

        // Prepare shape and then allocate tensor.
        jsize                shapeSize = env->GetArrayLength(dims);
        jlong *              arr       = env->GetLongArrayElements(dims, 0);
        std::vector<int64_t> shapes(arr, arr + shapeSize);
        env->ReleaseLongArrayElements(dims, arr, JNI_ABORT);

        auto  stringTensor = allocator->allocate_tensor<std::string>(shapes);
        jsize size         = env->CallIntMethod(data, njni::java_util_ArrayList_size);
        auto  flatAccessor = stringTensor->flat();
        for (jsize i = 0; i < size; ++i)
        {
            jstring element = static_cast<jstring>(env->CallObjectMethod(data, njni::java_util_ArrayList_get, i));
            flatAccessor[i] = njni::to_string(env, element);
            env->DeleteLocalRef(element);
        }

        return reinterpret_cast<jlong>(njni::toHeap(stringTensor));
    }
    catch (const std::exception &e)
    {
        njni::throw_java_exception(env, e.what());
    }
    return reinterpret_cast<jlong>(nullptr);
}
