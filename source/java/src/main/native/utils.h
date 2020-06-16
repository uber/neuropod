#pragma once

#include <neuropod/neuropod.hh>

#include <string>
#include <vector>

#include <jni.h>

namespace neuropod
{
namespace jni
{

// Convert jstring to cpp string
std::string toString(JNIEnv *env, jstring target);

// Get the default input dim from the tensor spec. -2 and -1 are treated as 1.
std::vector<int64_t> getDefaultInputDim(const TensorSpec &tensorSpec);

// Copy the data in java memory to tensor
std::shared_ptr<NeuropodValue> createTesnorFromJavaMemory(std::shared_ptr<NeuropodTensorAllocator> allocator,
                                                          JNIEnv *                                 env,
                                                          jobject                                  value,
                                                          TensorType                               type,
                                                          const std::vector<int64_t> &             dims);

// A wrapper for env->FindClass, will throw a cpp exception if the find fails.
jclass findClass(JNIEnv *env, const char *name);

// A wrapper for env->GetMethodID, will throw a cpp exception if the get fails.
jmethodID getMethodID(JNIEnv *env, jclass clazz, const char *name, const char *sig);

// A wrapper for env->GetStaticMethodID, will throw a cpp exception if the get fails.
jmethodID getStaticMethodID(JNIEnv *env, jclass clazz, const char *name, const char *sig);

// Get the name of a Java class
std::string getJclassName(JNIEnv *env, jclass clazz);

// Get the corresponding Java enum of org/neuropod/DataType, used in converting cpp enum to java enum.
jobject getFieldObject(JNIEnv *env, jclass dataTypes, std::string fieldName);

// Convert cpp tensor type to string
std::string tensorTypeToString(TensorType type);

// Throw a Java NeuropodJNIException exception. This throw is only valid for the Java side, in the cpp side it is just
// like a normal function: it won't interupt the normal program work flow.
void throwJavaException(JNIEnv *env, const char *message);

} // namespace jni
} // namespace neuropod
