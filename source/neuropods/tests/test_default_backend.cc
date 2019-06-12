//
// Uber, Inc. (c) 2019
//

#include "test_utils.hh"

// We need to override the default backend lookup paths
// Modifying the RPATH doesn't work on Linux because Bazel symlinks all the objects
// into the test directory
const std::unordered_map<std::string, std::string> default_backend_overrides = {
    {"tensorflow", "./neuropods/backends/tensorflow/libneuropod_tensorflow_backend.so"},
    {"python", "./neuropods/backends/python_bridge/libneuropod_pythonbridge_backend.so"},
    {"pytorch", "./neuropods/backends/python_bridge/libneuropod_pythonbridge_backend.so"},
    {"torchscript", "./neuropods/backends/torchscript/libneuropod_torchscript_backend.so"},
};

TEST(test_default_backend, test_pytorch_addition_model)
{
    // Test the PyTorch addition model
    test_addition_model("neuropods/tests/test_data/pytorch_addition_model/", default_backend_overrides);
}

TEST(test_default_backend, test_pytorch_strings_model)
{
    // Test the PyTorch strings model
    test_strings_model("neuropods/tests/test_data/pytorch_strings_model/", default_backend_overrides);
}

TEST(test_default_backend, test_torchscript_addition_model)
{
    // Test the TorchScript addition model
    test_addition_model("neuropods/tests/test_data/torchscript_addition_model/", default_backend_overrides);
}

TEST(test_default_backend, test_torchscript_strings_model)
{
    // Test the TorchScript strings model
    test_strings_model("neuropods/tests/test_data/torchscript_strings_model/", default_backend_overrides);
}

TEST(test_default_backend, test_tensorflow_addition_model)
{
    // Test the TensorFlow addition model
    test_addition_model("neuropods/tests/test_data/tf_addition_model/", default_backend_overrides);
}

TEST(test_default_backend, test_tensorflow_strings_model)
{
    // Test the TensorFlow strings model
    test_strings_model("neuropods/tests/test_data/tf_strings_model/", default_backend_overrides);
}
