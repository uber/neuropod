#include "jclass_register.h"

#include "utils.h"

#include <exception>
#include <iostream>

#include <jni.h>

using namespace neuropod::jni;

jclass    java_util_Map;
jmethodID java_util_Map_get;

jclass    java_util_ArrayList;
jmethodID java_util_ArrayList_;
jmethodID java_util_ArrayList_add;
jmethodID java_util_ArrayList_get;
jmethodID java_util_ArrayList_size;

jclass    java_lang_Float;
jmethodID java_lang_Float_valueOf;
jmethodID java_lang_Float_floatValue;

jclass    java_lang_Double;
jmethodID java_lang_Double_valueOf;
jmethodID java_lang_Double_doubleValue;

jclass    java_lang_Integer;
jmethodID java_lang_Integer_valueOf;
jmethodID java_lang_Integer_intValue;

jclass    java_lang_Long;
jmethodID java_lang_Long_valueOf;
jmethodID java_lang_Long_longValue;

jclass org_neuropod_NeuropodJNIException;

jclass org_neuropod_DataType;

jint JNI_VERSION = JNI_VERSION_1_8;

// This function is called when the JNI is loaded.
jint JNI_OnLoad(JavaVM *vm, void *reserved)
{

    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv *env;
    std::cout << "Reading" << std::endl;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION) != JNI_OK)
    {
        return JNI_ERR;
    }

    // Move this exception class out of try catch block to avoid unexpected error when throw a java exception and the
    // exception type is wrong
    org_neuropod_NeuropodJNIException =
        static_cast<jclass>(env->NewGlobalRef(findClass(env, "org/neuropod/NeuropodJNIException")));
    try
    {
        java_util_Map     = static_cast<jclass>(env->NewGlobalRef(findClass(env, "java/util/Map")));
        java_util_Map_get = getMethodID(env, java_util_Map, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");

        java_util_ArrayList        = static_cast<jclass>(env->NewGlobalRef(findClass(env, "java/util/ArrayList")));
        java_util_ArrayList_       = getMethodID(env, java_util_ArrayList, "<init>", "(I)V");
        java_util_ArrayList_add    = getMethodID(env, java_util_ArrayList, "add", "(Ljava/lang/Object;)Z");
        java_util_ArrayList_get    = getMethodID(env, java_util_ArrayList, "get", "(I)Ljava/lang/Object;");
        java_util_ArrayList_size   = getMethodID(env, java_util_ArrayList, "size", "()I");
        java_lang_Float            = static_cast<jclass>(env->NewGlobalRef(findClass(env, "java/lang/Float")));
        java_lang_Float_valueOf    = getStaticMethodID(env, java_lang_Float, "valueOf", "(F)Ljava/lang/Float;");
        java_lang_Float_floatValue = getMethodID(env, java_lang_Float, "floatValue", "()F");

        java_lang_Double             = static_cast<jclass>(env->NewGlobalRef(findClass(env, "java/lang/Double")));
        java_lang_Double_valueOf     = getStaticMethodID(env, java_lang_Double, "valueOf", "(D)Ljava/lang/Double;");
        java_lang_Double_doubleValue = getMethodID(env, java_lang_Double, "doubleValue", "()D");

        java_lang_Integer          = static_cast<jclass>(env->NewGlobalRef(findClass(env, "java/lang/Integer")));
        java_lang_Integer_valueOf  = getStaticMethodID(env, java_lang_Integer, "valueOf", "(I)Ljava/lang/Integer;");
        java_lang_Integer_intValue = getMethodID(env, java_lang_Integer, "intValue", "()I");

        java_lang_Long           = static_cast<jclass>(env->NewGlobalRef(findClass(env, "java/lang/Long")));
        java_lang_Long_valueOf   = getStaticMethodID(env, java_lang_Long, "valueOf", "(J)Ljava/lang/Long;");
        java_lang_Long_longValue = getMethodID(env, java_lang_Long, "longValue", "()J");

        org_neuropod_DataType = static_cast<jclass>(env->NewGlobalRef(findClass(env, "org/neuropod/DataType")));
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, e.what());
    }
    // Return the JNI Version as required by method
    return JNI_VERSION;
}

// This function is called when the JNI is unloaded.
void JNI_OnUnload(JavaVM *vm, void *reserved)
{

    // Obtain the JNIEnv from the VM
    // NOTE: some re-do the JNI Version check here, but I find that redundant
    JNIEnv *env;
    vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION);
    // Destroy the global references
    env->DeleteGlobalRef(java_util_Map);
    env->DeleteGlobalRef(java_util_ArrayList);
    env->DeleteGlobalRef(java_lang_Float);
    env->DeleteGlobalRef(java_lang_Double);
    env->DeleteGlobalRef(java_lang_Integer);
    env->DeleteGlobalRef(java_lang_Long);
    env->DeleteGlobalRef(org_neuropod_NeuropodJNIException);
}
