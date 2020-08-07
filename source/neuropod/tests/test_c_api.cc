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

#include "gtest/gtest.h"
#include "neuropod/bindings/c/c_api.h"

#include <string>
#include <vector>

TEST(test_c_api, basic)
{
    NP_Neuropod *model;
    NP_Status *  status = NP_NewStatus();

    // Load a model from a wrong path.
    NP_LoadNeuropod("wrong_path", &model, status);
    if (NP_GetCode(status) != NEUROPOD_ERROR)
    {
        // Throw an error here
        FAIL() << "Error from C API expeted error during model loading: " << NP_GetMessage(status);
    }

    // Load a model and get an allocator.
    NP_LoadNeuropod("neuropod/tests/test_data/tf_addition_model/", &model, status);
    if (NP_GetCode(status) != NEUROPOD_OK)
    {
        FAIL() << "Error from C API during model loading: " << NP_GetMessage(status);
    }

    // Test model name.
    if (NP_GetName(model) != std::string("addition_model"))
    {
        FAIL() << "Error from C API unexpected model name: " << NP_GetName(model);
    }

    // Test model platform.
    if (NP_GetPlatform(model) != std::string("tensorflow"))
    {
        FAIL() << "Error from C API unexpected model name: " << NP_GetPlatform(model);
    }

    NP_TensorAllocator *allocator = NP_GetAllocator(model);

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
    if (NP_GetCode(status) != NEUROPOD_ERROR)
    {
        FAIL() << "Error from C API error is expected during inference: " << NP_GetMessage(status);
    }

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
    if (NP_GetCode(status) != NEUROPOD_ERROR)
    {
        FAIL() << "Error from C API during inference: " << NP_GetMessage(status);
    }

    // Run succcessful inference
    NP_Infer(model, inputs, &outputs, status);
    if (NP_GetCode(status) != NEUROPOD_OK)
    {
        FAIL() << "Error from C API during inference: " << NP_GetMessage(status);
    }

    // The same inference but specify requested output.
    const char *requested_output[] = {"out"};
    NP_InferWithRequestedOutputs(model, inputs, 1, requested_output, &outputs, status);
    if (NP_GetCode(status) != NEUROPOD_OK)
    {
        FAIL() << "Error from C API during inference: " << NP_GetMessage(status);
    }

    // Test model input and output configuration.
    EXPECT_EQ(2, NP_GetNumInputs(model));
    EXPECT_EQ(1, NP_GetNumOutputs(model));

    // Get the output and compare to the expected value
    NP_NeuropodTensor *out       = NP_GetTensor(outputs, "out");
    float *            out_data  = reinterpret_cast<float *>(NP_GetData(out));
    size_t             nout_data = NP_GetNumElements(out);
    EXPECT_TRUE(std::equal(out_data, out_data + nout_data, target));

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
