//
// Uber, Inc. (c) 2018
//

#include "neuropods/tests/test_utils.hh"

TEST(test_models, test_tensorflow_addition_model)
{
    // Test the TensorFlow addition model using the native TensorFlow backend
    test_addition_model("neuropods/tests/test_data/tf_addition_model/", "TensorflowNeuropodBackend");
}
