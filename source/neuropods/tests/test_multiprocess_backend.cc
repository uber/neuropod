//
// Uber, Inc. (c) 2019
//

#include "neuropods/tests/test_utils.hh"

TEST(test_multiprocess_backend, test_pytorch_addition_model)
{
    // Test the PyTorch addition model using the multiprocess backend
    test_addition_model("neuropods/tests/test_data/pytorch_addition_model/", "MultiprocessNeuropodBackend");
}

TEST(test_multiprocess_backend, test_torchscript_addition_model)
{
    // Test the TorchScript addition model using the multiprocess backend
    test_addition_model("neuropods/tests/test_data/torchscript_addition_model/", "MultiprocessNeuropodBackend");
}

TEST(test_multiprocess_backend, test_tensorflow_addition_model)
{
    // Test the TensorFlow addition model using the multiprocess backend
    test_addition_model("neuropods/tests/test_data/tf_addition_model/", "MultiprocessNeuropodBackend");
}
