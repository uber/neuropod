#include "org_neuropod_NeuropodValue.h"

#include "jclass_register.h"
#include "utils.h"

#include <neuropod/neuropod.hh>

#include <string>

#include <jni.h>

using namespace neuropod::jni;

JNIEXPORT jobject JNICALL Java_org_neuropod_NeuropodValue_nativeToList(JNIEnv *env, jclass, jlong nativeHandle)
{
    // This function copy the data twice, first from neuropod to cpp vector, then from cpp vector to java list.
    // This is not very efficient.
    try
    {
        auto    neuropodTensor = reinterpret_cast<neuropod::NeuropodValue *>(nativeHandle)->as_tensor();
        auto    size           = neuropodTensor->get_num_elements();
        auto    tensorType     = neuropodTensor->get_tensor_type();
        jobject ret            = env->NewObject(java_util_ArrayList, java_util_ArrayList_, size);

        switch (tensorType)
        {
        case neuropod::FLOAT_TENSOR: {
            auto typedTensor = neuropodTensor->as_typed_tensor<float>();
            auto elementList = typedTensor->get_data_as_vector();
            for (auto ele : elementList)
            {
                jobject javaEle = env->CallStaticObjectMethod(java_lang_Float, java_lang_Float_valueOf, ele);
                env->CallBooleanMethod(ret, java_util_ArrayList_add, javaEle);
                env->DeleteLocalRef(javaEle);
            }
            break;
        }
        case neuropod::DOUBLE_TENSOR: {
            auto typedTensor = neuropodTensor->as_typed_tensor<double>();
            auto elementList = typedTensor->get_data_as_vector();
            for (auto ele : elementList)
            {
                jobject javaEle = env->CallStaticObjectMethod(java_lang_Double, java_lang_Double_valueOf, ele);
                env->CallBooleanMethod(ret, java_util_ArrayList_add, javaEle);
                env->DeleteLocalRef(javaEle);
            }
            break;
        }
        case neuropod::INT16_TENSOR:
        case neuropod::INT32_TENSOR: {
            auto typedTensor = neuropodTensor->as_typed_tensor<int32_t>();
            auto elementList = typedTensor->get_data_as_vector();
            for (auto ele : elementList)
            {
                jobject javaEle = env->CallStaticObjectMethod(java_lang_Integer, java_lang_Integer_valueOf, ele);
                env->CallBooleanMethod(ret, java_util_ArrayList_add, javaEle);
                env->DeleteLocalRef(javaEle);
            }
            break;
        }
        case neuropod::INT64_TENSOR: {
            auto typedTensor = neuropodTensor->as_typed_tensor<int64_t>();
            auto elementList = typedTensor->get_data_as_vector();
            for (auto ele : elementList)
            {
                jobject javaEle = env->CallStaticObjectMethod(java_lang_Long, java_lang_Long_valueOf, ele);
                env->CallBooleanMethod(ret, java_util_ArrayList_add, javaEle);
                env->DeleteLocalRef(javaEle);
            }
            break;
        }
        default:
            throw std::runtime_error(std::string("Unsupported tensor type:") + tensorTypeToString(tensorType));
        }
        return ret;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    return nullptr;
}

JNIEXPORT void JNICALL Java_org_neuropod_NeuropodValue_nativeDelete(JNIEnv *, jobject, jlong nativeHandle)
{
    auto pointer = reinterpret_cast<neuropod::NeuropodValue *>(nativeHandle);
    delete pointer;
}