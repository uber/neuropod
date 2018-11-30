//
// Uber, Inc. (c) 2018
//

#include "gtest/gtest.h"

#include "neuropods/backends/python_bridge/python_bridge.hh"
#include "neuropods/internal/tensor_store.hh"
#include "neuropods/neuropod_input_builder.hh"
#include "neuropods/neuropod_output_data.hh"


TEST(test_python_bridge, test_addition_model)
{
    // Some sample input data
    std::vector<int64_t> shape = {2, 2};

    float x_data[] = {1, 2, 3, 4};
    float y_data[] = {7, 8, 9, 10};
    float target[] = {8, 10, 12, 14};

    // Load a neuropod using the PythonBridge backend
    auto backend = std::make_shared<neuropods::PythonBridge>("neuropods/tests/test_data/pytorch_addition_model/");

    // Create the input builder and add some data
    neuropods::NeuropodInputBuilder builder(backend);
    auto in_store = builder.add_tensor("x", x_data, 4, shape).add_tensor("y", y_data, 4, shape).build();

    // Run inference
    auto out_store = backend->infer(*in_store);

    // Wrap the output in a NeuropodOutputData so we can read it
    // easily
    neuropods::NeuropodOutputData output_data(std::move(out_store));

    // Get the data in the output tensor
    std::vector<float>   out_vector = output_data.get_data_as_vector<float>("out");
    std::vector<int64_t> out_shape  = output_data.get_shape("out");

    // Check that the output data matches
    EXPECT_EQ(out_vector.size(), 4);
    EXPECT_EQ(memcmp(out_vector.data(), target, 4 * sizeof(float)), 0);

    // Check that the shape matches
    EXPECT_EQ(out_shape.size(), shape.size());
    EXPECT_EQ(memcmp(out_shape.data(), shape.data(), shape.size() * sizeof(float)), 0);
}
