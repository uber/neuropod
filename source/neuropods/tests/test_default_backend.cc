//
// Uber, Inc. (c) 2019
//

#include "test_utils.hh"

TEST(test_default_backend, test_pytorch_addition_model)
{
    // Test the PyTorch addition model
    test_addition_model("neuropods/tests/test_data/pytorch_addition_model/");
}

TEST(test_default_backend, test_pytorch_strings_model)
{
    // Test the PyTorch strings model
    test_strings_model("neuropods/tests/test_data/pytorch_strings_model/");
}

TEST(test_default_backend, test_torchscript_addition_model)
{
    // Test the TorchScript addition model
    test_addition_model("neuropods/tests/test_data/torchscript_addition_model/");
}

TEST(test_default_backend, test_torchscript_strings_model)
{
    // Test the TorchScript strings model
    test_strings_model("neuropods/tests/test_data/torchscript_strings_model/");
}

TEST(test_default_backend, test_tensorflow_addition_model)
{
    // Test the TensorFlow addition model
    test_addition_model("neuropods/tests/test_data/tf_addition_model/");
}

TEST(test_default_backend, test_tensorflow_strings_model)
{
    // Test the TensorFlow strings model
    test_strings_model("neuropods/tests/test_data/tf_strings_model/");
}
