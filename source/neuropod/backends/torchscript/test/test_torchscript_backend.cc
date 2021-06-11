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

#include "neuropod/tests/test_utils.hh"

TEST(test_torchscript_backend, test_torchscript_addition_model)
{
    // Test the TorchScript addition model using the native torchscript backend
    test_addition_model("neuropod/tests/test_data/torchscript_addition_model/");
}

TEST(test_torchscript_backend, test_torchscript_addition_tensor_output_model)
{
    // Test the TorchScript addition model using the native torchscript backend
    test_addition_model("neuropod/tests/test_data/torchscript_addition_model_single_output/");
}

TEST(test_torchscript_backend, test_torchscript_strings_model)
{
    // Test the TorchScript strings model using the native torchscript backend
    test_strings_model("neuropod/tests/test_data/torchscript_strings_model/");
}

TEST(test_torchscript_backend, invalid_dtype)
{
    neuropod::Neuropod model("neuropod/tests/test_data/torchscript_strings_model/");

    // These should work
    model.allocate_tensor<float>({2});
    model.allocate_tensor<uint8_t>({2});

    // These types aren't supported by torch
    EXPECT_THROW(model.allocate_tensor<uint16_t>({2}), std::runtime_error);
    EXPECT_THROW(model.allocate_tensor<uint32_t>({2}), std::runtime_error);
    EXPECT_THROW(model.allocate_tensor<uint64_t>({2}), std::runtime_error);
}

// TODO(vip): reenable this test once we have more complete support for directly loading models
// TEST(test_torchscript_backend, load_model_from_path)
// {
//     // Load a TorchScript model directly
//     neuropod::TorchNeuropodBackend model("neuropod/tests/test_data/torchscript_strings_model/0/data/model.pt");
// }

TEST(test_multiprocess_backend, test_torchscript_addition_model)
{
    // Test the TorchScript addition model in another process
    test_addition_model_ope("neuropod/tests/test_data/torchscript_addition_model/");
}

TEST(test_multiprocess_backend, test_torchscript_strings_model)
{
    // Test the TorchScript strings model in another process
    test_strings_model_ope("neuropod/tests/test_data/torchscript_strings_model/");
}