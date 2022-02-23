#include "com_uber_neuropod_NeuropodTensor.h"

#include "jclass_register.h"
#include "neuropod/neuropod.hh"
#include "utils.h"

#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <jni.h>

namespace njni = neuropod::jni;

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for Java API methods
JNIEXPORT void JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeDoDelete(JNIEnv *env,
                                                                            jobject /*unused*/,
                                                                            jlong handle)
{
    try
    {
        // Java NeuropodTensor stores a pointer to a shared_ptr
        auto tensorPtr = reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(handle);
        std::unique_ptr<std::shared_ptr<neuropod::NeuropodValue>> scopeHolder(tensorPtr);
        tensorPtr->reset();
    }
    catch (const std::exception &e)
    {
        njni::throw_java_exception(env, e.what());
    }
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for Java API methods
JNIEXPORT jobject JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeGetBuffer(JNIEnv *env,
                                                                                jclass /*unused*/,
                                                                                jlong nativeHandle)
{
    try
    {
        auto neuropodTensor =
            (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(nativeHandle))->as_tensor();
        auto tensorType = neuropodTensor->get_tensor_type();
        switch (tensorType)
        {
        case neuropod::FLOAT_TENSOR: {
            return njni::createDirectBuffer<float>(env, neuropodTensor);
        }
        case neuropod::DOUBLE_TENSOR: {
            return njni::createDirectBuffer<double>(env, neuropodTensor);
        }
        case neuropod::INT32_TENSOR: {
            return njni::createDirectBuffer<int32_t>(env, neuropodTensor);
        }
        case neuropod::INT64_TENSOR: {
            return njni::createDirectBuffer<int64_t>(env, neuropodTensor);
        }
        case neuropod::STRING_TENSOR: {
            // If it is STRING_TENSOR, we would flatten the tensor data and convert it to a string list
            // we don't need the buffer to store the data
            return env->NewGlobalRef(NULL);
        }
        default:
            throw std::runtime_error("unsupported tensor type: " + njni::tensor_type_to_string(tensorType));
        }
    }
    catch (const std::exception &e)
    {
        njni::throw_java_exception(env, e.what());
    }
    return nullptr;
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for Java API methods
JNIEXPORT jlongArray JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeGetDims(JNIEnv *env,
                                                                                 jclass /*unused*/,
                                                                                 jlong handle)
{
    try
    {
        auto       tensor = (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(handle))->as_tensor();
        auto       dims   = tensor->as_tensor()->get_dims();
        jlongArray ret    = env->NewLongArray(dims.size());
        if (!ret || env->ExceptionCheck())
        {
            throw std::runtime_error("NewLongArray failed: cannot allocate long array");
        }
        env->SetLongArrayRegion(ret, 0, dims.size(), reinterpret_cast<jlong *>(dims.data()));
        return ret;
    }
    catch (const std::exception &e)
    {
        njni::throw_java_exception(env, e.what());
    }
    return nullptr;
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for Java API methods
JNIEXPORT jobject JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeGetTensorType(JNIEnv *env,
                                                                                    jclass /*unused*/,
                                                                                    jlong handle)
{
    try
    {
        auto tensor = (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(handle))->as_tensor();
        auto type   = tensor->as_tensor()->get_tensor_type();
        return njni::get_tensor_type_field(env, njni::tensor_type_to_string(type));
    }
    catch (const std::exception &e)
    {
        njni::throw_java_exception(env, e.what());
    }
    return nullptr;
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for Java API methods
JNIEXPORT jlong JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeGetNumberOfElements(JNIEnv *env,
                                                                                        jclass /*unused*/,
                                                                                        jlong handle)
{
    try
    {
        auto tensor = (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(handle))->as_tensor();
        return tensor->get_num_elements();
    }
    catch (const std::exception &e)
    {
        njni::throw_java_exception(env, e.what());
    }
    return 0;
}

JNIEXPORT jobject JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeToStringList(JNIEnv *env, jclass, jlong handle)
{
    try
    {
        auto stringTensor = (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(handle))
                                ->as_tensor()
                                ->as_typed_tensor<std::string>();
        auto    size = stringTensor->get_num_elements();
        jobject ret  = env->NewObject(njni::java_util_ArrayList, njni::java_util_ArrayList_, size);
        if (!ret || env->ExceptionCheck())
        {
            throw std::runtime_error("NewObject failed: cannot create ArrayList");
        }

        auto flatAccessor = stringTensor->flat();
        for (size_t i = 0; i < size; ++i)
        {
            const std::string &elem          = flatAccessor[i];
            jstring            convertedElem = njni::to_jstring(env, elem);
            env->CallBooleanMethod(ret, njni::java_util_ArrayList_add, convertedElem);
            env->DeleteLocalRef(convertedElem);
        }

        return ret;
    }
    catch (const std::exception &e)
    {
        njni::throw_java_exception(env, e.what());
    }
    return nullptr;
}

JNIEXPORT jstring JNICALL Java_com_uber_neuropod_NeuropodTensor_nativeGetString(JNIEnv *env,
                                                                                jclass,
                                                                                jlong index,
                                                                                jlong handle)
{
    try
    {
        auto stringTensor = (*reinterpret_cast<std::shared_ptr<neuropod::NeuropodValue> *>(handle))
                                ->as_tensor()
                                ->as_typed_tensor<std::string>();
        const std::string &elem = stringTensor->flat()[index];
        return njni::to_jstring(env, elem);
    }
    catch (const std::exception &e)
    {
        njni::throw_java_exception(env, e.what());
    }
    return nullptr;
}
