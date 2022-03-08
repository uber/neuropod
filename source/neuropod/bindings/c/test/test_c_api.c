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

// Inspired by the TensorFlow C API c_test.

#include "neuropod/bindings/c/c_api.h"

// This file exists just to verify that the header files above can build,
// link, and run as "C" code.

#ifdef __cplusplus
#error "This file should be compiled as C code, not as C++."
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// gtest is used in C++ tests but it doesn't have C interfaces. Instead use
// ASSERT helpers implemented in Tensorflow C test to address the same problem
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/c_test.c
static void CheckFailed(const char *expression, const char *filename, int line_number)
{
    fprintf(stderr, "ERROR: CHECK failed: %s:%d: %s\n", filename, line_number, expression);
    fflush(stderr);
    abort();
}

// We use an extra level of macro indirection here to ensure that the
// macro arguments get evaluated, so that in a call to CHECK(foo),
// the call to STRINGIZE(condition) in the definition of the CHECK
// macro results in the string "foo" rather than the string "condition".
#define STRINGIZE(expression) STRINGIZE2(expression)
#define STRINGIZE2(expression) #expression

// Like assert(), but not dependent on NDEBUG.
#define CHECK(condition) ((condition) ? (void) 0 : CheckFailed(STRINGIZE(condition), __FILE__, __LINE__))
#define ASSERT_EQ(expected, actual) CHECK((expected) == (actual))
#define ASSERT_NE(expected, actual) CHECK((expected) != (actual))
#define ASSERT_STREQ(expected, actual) ASSERT_EQ(0, strcmp((expected), (actual)))
#define ASSERT_ARRAYEQ(expected, actual, size) ASSERT_EQ(0, memcmp((expected), (actual), size))

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
static void TestLoadAndInference(void)
{
    NP_Neuropod *model = NULL;

    NP_Status *status = NP_NewStatus();

    // Load a model from a wrong path.
    NP_LoadNeuropod("wrong_path", &model, status);
    ASSERT_EQ(NP_GetCode(status), NEUROPOD_ERROR);
    // Note that on this failure model is not null,
    // because neuropod object was allocated.
    ASSERT_NE(model, NULL);

    // Free the model we loaded with error.
    NP_FreeNeuropod(model);

    // Load a model and get an allocator.
    NP_LoadNeuropod("neuropod/tests/test_data/tf_addition_model/", &model, status);
    ASSERT_EQ(NP_GetCode(status), NEUROPOD_OK);
    ASSERT_NE(model, NULL);

    // Test model name.
    ASSERT_STREQ(NP_GetName(model), "addition_model");

    // Test model platform.
    ASSERT_STREQ(NP_GetPlatform(model), "tensorflow");

    NP_TensorAllocator *allocator = NP_GetAllocator(model);
    ASSERT_NE(allocator, NULL);

    // Create tensors
    int64_t            dims[] = {2, 2};
    NP_NeuropodTensor *x      = NP_AllocateTensor(allocator, sizeof(dims) / sizeof(int64_t), dims, FLOAT_TENSOR);
    NP_NeuropodTensor *y      = NP_AllocateTensor(allocator, sizeof(dims) / sizeof(int64_t), dims, FLOAT_TENSOR);

    // Copy in data
    const float x_data[] = {1, 2, 3, 4};
    const float y_data[] = {7, 8, 9, 10};
    const float target[] = {8, 10, 12, 14};
    memcpy(NP_GetData(x), x_data, sizeof(x_data));
    memcpy(NP_GetData(y), y_data, sizeof(y_data));

    // Create the input
    NP_NeuropodValueMap *inputs = NP_NewValueMap();
    NP_NeuropodValueMap *outputs;

    // Run inference with empty input that should fail.
    NP_Infer(model, inputs, &outputs, status);
    ASSERT_EQ(NP_GetCode(status), NEUROPOD_ERROR);

    // Insert tensors into inputs
    NP_InsertTensor(inputs, "x", x);
    NP_InsertTensor(inputs, "y", y);

    ASSERT_EQ(NP_GetType(x), FLOAT_TENSOR);
    ASSERT_EQ(NP_GetType(y), FLOAT_TENSOR);

    // Free the input tensors
    NP_FreeTensor(x);
    NP_FreeTensor(y);

    // Free the allocator
    NP_FreeAllocator(allocator);

    // Run succcessful inference.
    NP_Infer(model, inputs, &outputs, status);
    ASSERT_EQ(NP_GetCode(status), NEUROPOD_OK);

    // Test model input and output configuration.
    ASSERT_EQ(2, NP_GetNumInputs(model));
    ASSERT_EQ(1, NP_GetNumOutputs(model));

    // Get the output and compare to the expected value
    NP_NeuropodTensor *out       = NP_GetTensor(outputs, "out");
    float *            out_data  = (float *) NP_GetData(out);
    size_t             nout_data = NP_GetNumElements(out);

    ASSERT_EQ(NP_GetType(out), FLOAT_TENSOR);

    for (size_t i = 0; i < nout_data; ++i)
    {
        ASSERT_EQ(out_data[i], target[i]);
    }

    // Free the input and output maps
    NP_FreeValueMap(inputs);
    NP_FreeValueMap(outputs);

    // Decrement the output tensor refcount
    NP_FreeTensor(out);

    // Delete the status
    NP_DeleteStatus(status);

    // Free the model we loaded
    NP_FreeNeuropod(model);
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
static void TestLoadAndInferenceWithOptions(void)
{
    NP_Neuropod *model = NULL;

    NP_Status *status = NP_NewStatus();

    // Load a model with runtime options. Change defaults.
    NP_RuntimeOptions opts                   = NP_DefaultRuntimeOptions();
    opts.use_ope                             = true;
    opts.ope_options.free_memory_every_cycle = true;
    opts.visible_device                      = GPU7;
    NP_LoadNeuropodWithOpts("neuropod/tests/test_data/tf_addition_model/", &opts, &model, status);
    ASSERT_EQ(NP_GetCode(status), NEUROPOD_OK);
    ASSERT_NE(model, NULL);

    // Test model name.
    ASSERT_STREQ(NP_GetName(model), "addition_model");

    // Test model platform.
    ASSERT_STREQ(NP_GetPlatform(model), "tensorflow");

    NP_TensorAllocator *allocator = NP_GetAllocator(model);
    ASSERT_NE(allocator, NULL);

    // Create tensors
    int64_t            dims[] = {2, 2};
    NP_NeuropodTensor *x      = NP_AllocateTensor(allocator, sizeof(dims) / sizeof(int64_t), dims, FLOAT_TENSOR);
    NP_NeuropodTensor *y      = NP_AllocateTensor(allocator, sizeof(dims) / sizeof(int64_t), dims, FLOAT_TENSOR);

    // Copy in data
    const float x_data[] = {1, 2, 3, 4};
    const float y_data[] = {7, 8, 9, 10};
    const float target[] = {8, 10, 12, 14};
    memcpy(NP_GetData(x), x_data, sizeof(x_data));
    memcpy(NP_GetData(y), y_data, sizeof(y_data));

    // Create the input
    NP_NeuropodValueMap *inputs = NP_NewValueMap();
    NP_NeuropodValueMap *outputs;

    // Insert tensors into inputs
    NP_InsertTensor(inputs, "x", x);
    NP_InsertTensor(inputs, "y", y);

    // Free the input tensors
    NP_FreeTensor(x);
    NP_FreeTensor(y);

    // Free the allocator
    NP_FreeAllocator(allocator);

    // Test that wrong requested_output fails.
    const char *requested_output_wrong[] = {"out", "out_wrong"};
    NP_InferWithRequestedOutputs(model, inputs, 2, requested_output_wrong, &outputs, status);
    ASSERT_EQ(NP_GetCode(status), NEUROPOD_ERROR);

    // Run succcessful inference and  specify requested output.
    const char *requested_output[] = {"out"};
    NP_InferWithRequestedOutputs(model, inputs, 1, requested_output, &outputs, status);
    ASSERT_EQ(NP_GetCode(status), NEUROPOD_OK);

    // Test model input and output configuration.
    ASSERT_EQ(2, NP_GetNumInputs(model));
    ASSERT_EQ(1, NP_GetNumOutputs(model));

    // Get the output and compare to the expected value
    NP_NeuropodTensor *out       = NP_GetTensor(outputs, "out");
    float *            out_data  = (float *) NP_GetData(out);
    size_t             nout_data = NP_GetNumElements(out);
    for (size_t i = 0; i < nout_data; ++i)
    {
        ASSERT_EQ(out_data[i], target[i]);
    }

    // Free the input and output maps
    NP_FreeValueMap(inputs);
    NP_FreeValueMap(outputs);

    // Decrement the output tensor refcount
    NP_FreeTensor(out);

    // Delete the status
    NP_DeleteStatus(status);

    // Free the model we loaded
    NP_FreeNeuropod(model);
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
static void TestTensorGetters(void)
{
    NP_TensorAllocator *allocator = NP_GetGenericAllocator();
    ASSERT_NE(allocator, NULL);

    // Create tensors with different types and test it.
    int64_t            dims[] = {2, 2};
    NP_NeuropodTensor *x      = NP_AllocateTensor(allocator, sizeof(dims) / sizeof(int64_t), dims, FLOAT_TENSOR);
    NP_NeuropodTensor *y      = NP_AllocateTensor(allocator, sizeof(dims) / sizeof(int64_t), dims, DOUBLE_TENSOR);
    NP_NeuropodTensor *z      = NP_AllocateTensor(allocator, sizeof(dims) / sizeof(int64_t), dims, STRING_TENSOR);

    ASSERT_EQ(FLOAT_TENSOR, NP_GetType(x));
    ASSERT_EQ(DOUBLE_TENSOR, NP_GetType(y));
    ASSERT_EQ(STRING_TENSOR, NP_GetType(z));

    size_t         test_num_dims;
    const int64_t *test_dims;
    NP_GetDims(x, &test_num_dims, &test_dims);

    ASSERT_EQ(sizeof(dims) / sizeof(dims[0]), test_num_dims);
    ASSERT_ARRAYEQ(&dims[0], test_dims, test_num_dims);

    NP_FreeTensor(x);
    NP_FreeTensor(y);
    NP_FreeTensor(z);

    NP_FreeAllocator(allocator);
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
static void RunTests(void)
{
    TestLoadAndInference();
    TestLoadAndInferenceWithOptions();
    TestTensorGetters();
}

int main(void)
{
    RunTests();
    return 0;
}
