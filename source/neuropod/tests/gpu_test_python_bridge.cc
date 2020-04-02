//
// Uber, Inc. (c) 2019
//

#include "test_utils.hh"

#include <thread>

TEST(test_models, test_pytorch_addition_model)
{
    // Test the PyTorch addition model using the python bridge
    test_addition_model("neuropod/tests/test_data/pytorch_addition_model_gpu/");
}

TEST(test_models, test_pytorch_addition_model_threaded)
{
    std::thread t([]() {
        // Test the PyTorch addition model using the python bridge
        test_addition_model("neuropod/tests/test_data/pytorch_addition_model_gpu/");
    });

    t.join();
}
