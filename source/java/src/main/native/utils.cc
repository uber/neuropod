#include "utils.h"

#include "jclass_register.h"

#include <neuropod/neuropod.hh>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <jni.h>

namespace neuropod
{
namespace jni
{

const std::string NEUROPOD_JNI_EXCEPTION = "org/neuropod/NeuropodJNIException";

std::string toString(JNIEnv *env, jstring target)
{
    const char *raw = env->GetStringUTFChars(target, NULL);
    std::string res(raw);
    env->ReleaseStringUTFChars(target, raw);
    return res;
}

std::vector<int64_t> getDefaultInputDim(const TensorSpec &tensorSpec)
{
    std::vector<int64_t> ret;
    for (auto &dim : tensorSpec.dims)
    {
        if (dim.value == -1 || dim.value == -2)
        {
            ret.push_back(1);
        }
        else
        {
            ret.push_back(dim.value);
        }
    }
    return ret;
}

std::shared_ptr<NeuropodValue> createTesnorFromJavaMemory(std::shared_ptr<NeuropodTensorAllocator> allocator,
                                                          JNIEnv *                                 env,
                                                          jobject                                  value,
                                                          TensorType                               type,
                                                          const std::vector<int64_t> &             dims)
{
    auto  tensor = allocator->allocate_tensor(dims, type);
    jlong size   = env->CallIntMethod(value, java_util_ArrayList_size);
    // Here we copy the data twice: First from java list to cpp vector, then the neuropod copy the vector to
    // its own data type. This is not very efficient.
    switch (type)
    {
    case FLOAT_TENSOR: {
        std::vector<float> elementList(size);
        for (int i = 0; i < size; i++)
        {
            jobject element = env->CallObjectMethod(value, java_util_ArrayList_get, i);
            elementList[i]  = env->CallFloatMethod(element, java_lang_Float_floatValue);
            env->DeleteLocalRef(element);
        }
        auto floatTensor = tensor->as_typed_tensor<float>();
        floatTensor->copy_from(elementList);
        break;
    }
    case DOUBLE_TENSOR: {
        std::vector<double> elementList(size);
        for (int i = 0; i < size; i++)
        {
            jobject element = env->CallObjectMethod(value, java_util_ArrayList_get, i);
            elementList[i]  = env->CallDoubleMethod(element, java_lang_Double_doubleValue);
            env->DeleteLocalRef(element);
        }
        auto doubleTensor = tensor->as_typed_tensor<double>();
        doubleTensor->copy_from(elementList);
        break;
    }
    case INT32_TENSOR: {
        std::vector<int32_t> elementList(size);
        for (int i = 0; i < size; i++)
        {
            jobject element = env->CallObjectMethod(value, java_util_ArrayList_get, i);
            elementList[i]  = env->CallIntMethod(element, java_lang_Integer_intValue);
            env->DeleteLocalRef(element);
        }
        auto int32Tensor = tensor->as_typed_tensor<int32_t>();
        int32Tensor->copy_from(elementList);
        break;
    }
    case INT64_TENSOR: {
        std::vector<int64_t> elementList(size);
        for (int i = 0; i < size; i++)
        {
            jobject element = env->CallObjectMethod(value, java_util_ArrayList_get, i);
            elementList[i]  = env->CallIntMethod(element, java_lang_Long_longValue);
            env->DeleteLocalRef(element);
        }
        auto int64Tensor = tensor->as_typed_tensor<int64_t>();
        int64Tensor->copy_from(elementList);
        break;
    }
    case STRING_TENSOR: {
        std::vector<std::string> elementList(size);
        for (int i = 0; i < size; i++)
        {
            elementList[i] =
                toString(env, static_cast<jstring>(env->CallObjectMethod(value, java_util_ArrayList_get, i)));
        }
        auto stringTensor = tensor->as_typed_tensor<std::string>();
        stringTensor->copy_from(elementList);
        break;
    }
    default:
        throw std::runtime_error(std::string("Unsupported tensor type: ") + tensorTypeToString(type));
    }
    return tensor;
}

jclass findClass(JNIEnv *env, const char *name)
{
    jclass ret = env->FindClass(name);
    if (reinterpret_cast<jlong>(ret) == 0)
    {
        throw std::runtime_error(std::string("Class not found: ") + name);
    }
    return ret;
}

std::string getJclassName(JNIEnv *env, jclass clazz)
{
    // TODO: get class name from jclass by weijiad
    return "";
}

jmethodID getMethodID(JNIEnv *env, jclass clazz, const char *name, const char *sig)
{
    jmethodID ret = env->GetMethodID(clazz, name, sig);
    if (reinterpret_cast<jlong>(ret) == 0)
    {
        throw std::runtime_error(std::string("Method ID not found: ") + getJclassName(env, clazz) + name + sig);
    }
    return ret;
}

jmethodID getStaticMethodID(JNIEnv *env, jclass clazz, const char *name, const char *sig)
{
    jmethodID ret = env->GetStaticMethodID(clazz, name, sig);
    if (reinterpret_cast<jlong>(ret) == 0)
    {
        throw std::runtime_error(std::string("Method ID not found: ") + getJclassName(env, clazz) + name + sig);
    }
    return ret;
}

jobject getFieldObject(JNIEnv *env, jclass dataTypes, std::string fieldName)
{
    jfieldID field = env->GetStaticFieldID(dataTypes, fieldName.c_str(), "Lorg/neuropod/DataType;");
    if (reinterpret_cast<jlong>(field) == 0)
    {
        throw std::runtime_error(std::string("Field not found: ") + getJclassName(env, dataTypes) + fieldName);
    }
    return env->GetStaticObjectField(dataTypes, field);
}

std::string tensorTypeToString(TensorType type)
{
    std::string       typeString;
    std::stringstream ss;
    ss << type;
    ss >> typeString;
    return typeString;
}

void throwJavaException(JNIEnv *env, const char* message)
{
    env->ThrowNew(org_neuropod_NeuropodJNIException, message);
}

} // namespace jni
} // namespace neuropod