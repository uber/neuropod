//
// Uber, Inc. (c) 2018
//

#include "test_utils.hh"

TEST(test_torchscript_backend, test_torchscript_addition_model)
{
    // Test the TorchScript addition model using the native torchscript backend
    test_addition_model("neuropods/tests/test_data/torchscript_addition_model/", "TorchNeuropodBackend");
}

TEST(test_torchscript_backend, test_torchscript_addition_tensor_output_model)
{
    // Test the TorchScript addition model using the native torchscript backend
    test_addition_model("neuropods/tests/test_data/torchscript_addition_model_single_output/", "TorchNeuropodBackend");
}

TEST(test_torchscript_backend, test_torchscript_strings_model)
{
    // Test the TorchScript strings model using the native torchscript backend
    test_strings_model("neuropods/tests/test_data/torchscript_strings_model/", "TorchNeuropodBackend");
}

TEST(test_torchscript_backend, invalid_dtype)
{
    neuropods::Neuropod model("neuropods/tests/test_data/torchscript_strings_model/", "TorchNeuropodBackend");

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
//     neuropods::TorchNeuropodBackend model("neuropods/tests/test_data/torchscript_strings_model/0/data/model.pt");
// }
