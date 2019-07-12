//
// Uber, Inc. (c) 2018
//

#include "test_utils.hh"

TEST(test_models, test_torchscript_addition_model)
{
    // Test the TorchScript addition model using the native torchscript backend
    test_addition_model("neuropods/tests/test_data/torchscript_addition_model/", "TorchNeuropodBackend");
}

TEST(test_models, test_torchscript_addition_tensor_output_model)
{
    // Test the TorchScript addition model using the native torchscript backend
    test_addition_model("neuropods/tests/test_data/torchscript_addition_model_single_output/", "TorchNeuropodBackend");
}


TEST(test_models, test_torchscript_strings_model)
{
    // Test the TorchScript strings model using the native torchscript backend
    test_strings_model("neuropods/tests/test_data/torchscript_strings_model/", "TorchNeuropodBackend");
}
