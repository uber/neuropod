//
// Uber, Inc. (c) 2018
//

#include "test_utils.hh"

TEST(test_models, test_pytorch_addition_model)
{
    // Test the PyTorch addition model using the python bridge
    test_addition_model("neuropods/tests/test_data/pytorch_addition_model/", "PythonBridge");
}

TEST(test_models, test_pytorch_strings_model)
{
    // Test the PyTorch strings model using the python bridge
    test_strings_model("neuropods/tests/test_data/pytorch_strings_model/", "PythonBridge");
}

TEST(test_models, test_torchscript_addition_model)
{
    // Test the TorchScript addition model using the python bridge
    test_addition_model("neuropods/tests/test_data/torchscript_addition_model/", "PythonBridge");
}

TEST(test_models, test_torchscript_strings_model)
{
    // Test the TorchScript strings model using the python bridge
    test_strings_model("neuropods/tests/test_data/torchscript_strings_model/", "PythonBridge");
}
