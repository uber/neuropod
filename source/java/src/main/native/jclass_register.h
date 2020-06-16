#pragma once

#include <jni.h>

extern jclass    java_util_Map;
extern jmethodID java_util_Map_get;

extern jclass    java_util_ArrayList;
extern jmethodID java_util_ArrayList_;
extern jmethodID java_util_ArrayList_add;
extern jmethodID java_util_ArrayList_get;
extern jmethodID java_util_ArrayList_size;

extern jclass    java_lang_Float;
extern jmethodID java_lang_Float_valueOf;
extern jmethodID java_lang_Float_floatValue;

extern jclass    java_lang_Double;
extern jmethodID java_lang_Double_valueOf;
extern jmethodID java_lang_Double_doubleValue;

extern jclass    java_lang_Integer;
extern jmethodID java_lang_Integer_valueOf;
extern jmethodID java_lang_Integer_intValue;

extern jclass    java_lang_Long;
extern jmethodID java_lang_Long_valueOf;
extern jmethodID java_lang_Long_longValue;

extern jclass org_neuropod_NeuropodJNIException;

extern jclass org_neuropod_DataType;
