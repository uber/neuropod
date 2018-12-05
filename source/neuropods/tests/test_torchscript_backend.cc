//
// Uber, Inc. (c) 2018
//

#include "test_utils.hh"

TEST(test_models, test_torchscript_addition_model)
{
    // Test the TorchScript addition model using the native torchscript backend
    test_addition_model("neuropods/tests/test_data/torchscript_addition_model/", "TorchNeuropodBackend");
}
