/* Copyright (c) 2020 The Neuropod Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "neuropod/neuropod.hh"

#include <thread>

TEST(test_ope_multiple_instances, multithreaded)
{
    constexpr auto NUM_INSTANCES = 4;
    constexpr auto NUM_REQUESTS  = 2048;

    // Create models (on the main thread)
    std::vector<neuropod::Neuropod> models;
    for (size_t i = 0; i < NUM_INSTANCES; i++)
    {
        neuropod::RuntimeOptions opts;
        opts.use_ope = true;
        models.emplace_back("neuropod/tests/test_data/pytorch_strings_model/", opts);
    }

    // Start worker threads to run inference
    std::vector<std::thread> workers;
    workers.reserve(models.size());
    for (auto &model : models)
    {
        // Every thread uses a dedicated instance
        workers.emplace_back([&model]() {
            // Shape and target information to validate output
            const std::vector<int64_t>     shape  = {3};
            const std::vector<std::string> target = {"apple sauce", "banana pudding", "carrot cake"};

            for (size_t i = 0; i < NUM_REQUESTS; ++i)
            {
                auto x_ten = model.allocate_tensor<std::string>(shape);
                auto y_ten = model.allocate_tensor<std::string>(shape);

                x_ten->copy_from({"apple", "banana", "carrot"});
                y_ten->copy_from({"sauce", "pudding", "cake"});

                // Run inference
                const auto output_data = model.infer({{"x", x_ten}, {"y", y_ten}});

                // Check that the output data matches expected
                const auto out_tensor = output_data->at("out")->as_typed_tensor<std::string>();

                const auto  out_vector = out_tensor->get_data_as_vector();
                const auto &out_shape  = out_tensor->get_dims();

                // Check that the output data matches
                EXPECT_EQ(out_vector.size(), 3);
                EXPECT_TRUE(out_vector == target);

                // Check that the shape matches
                EXPECT_TRUE(out_shape == shape);
            }
        });
    }

    // Join threads
    std::for_each(workers.begin(), workers.end(), [](std::thread &t) { t.join(); });
}
