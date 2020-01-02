//
// Uber, Inc. (c) 2018
//

#include "neuropod/tests/test_utils.hh"

TEST(test_models, test_tensorflow_addition_model)
{
    // Test the TensorFlow addition model using the native TensorFlow backend
    test_addition_model("neuropod/tests/test_data/tf_addition_model/", "TensorflowNeuropodBackend");
}

TEST(test_models, test_tensorflow_strings_model)
{
    // Test the TensorFlow strings model using the native TensorFlow backend
    test_strings_model("neuropod/tests/test_data/tf_strings_model/", "TensorflowNeuropodBackend");
}
