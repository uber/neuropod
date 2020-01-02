//
// Uber, Inc. (c) 2019
//

#include "neuropod/multiprocess/multiprocess.hh"
#include "neuropod/tests/test_utils.hh"

TEST(test_multiprocess_backend, test_pytorch_addition_model)
{
    // Test the PyTorch addition model in another process
    auto neuropod = neuropod::load_neuropod_in_new_process("neuropod/tests/test_data/pytorch_addition_model/");
    test_addition_model(*neuropod);
}

TEST(test_multiprocess_backend, test_torchscript_addition_model)
{
    // Test the TorchScript addition model in another process
    auto neuropod = neuropod::load_neuropod_in_new_process("neuropod/tests/test_data/torchscript_addition_model/");
    test_addition_model(*neuropod);
}

TEST(test_multiprocess_backend, test_tensorflow_addition_model)
{
    // Test the TensorFlow addition model in another process
    auto neuropod = neuropod::load_neuropod_in_new_process("neuropod/tests/test_data/tf_addition_model/");
    test_addition_model(*neuropod);
}
