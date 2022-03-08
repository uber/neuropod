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

#include "gtest/gtest.h"
#include "neuropod/internal/config_utils.hh"

namespace
{

// UATG(clang-format/format) intentionally formatted
const std::string VALID_SPEC = "{"
                               "  \"name\": \"addition_model\","
                               "  \"platform\": \"tensorflow\","
                               "  \"input_spec\": [{"
                               "    \"dtype\": \"float32\","
                               "    \"shape\": [null, 2, \"some_symbol\"],"
                               "    \"name\": \"x\""
                               "  }],"
                               "  \"output_spec\": [{"
                               "    \"dtype\": \"float32\","
                               "    \"shape\": [null, 2, \"some_symbol\"],"
                               "    \"name\": \"y\""
                               "  }],"
                               "  \"input_tensor_device\": {"
                               "    \"x\": \"GPU\""
                               "  }"
                               "}";

std::string replace(const std::string &search, const std::string &replace)
{
    // Replace `search` with `replace` in the config above
    std::string spec = VALID_SPEC;
    return spec.replace(spec.find(search), search.length(), replace);
}

} // namespace

TEST(test_config_utils, valid_config)
{
    // Test that the config is valid
    std::istringstream config(VALID_SPEC);
    auto               model_config = neuropod::load_model_config(config);
    EXPECT_EQ("x", model_config->inputs[0].name);
    EXPECT_EQ(neuropod::TensorType::FLOAT_TENSOR, model_config->inputs[0].type);
    EXPECT_EQ(std::vector<neuropod::Dimension>({-1, 2, neuropod::Dimension("some_symbol")}),
              model_config->inputs[0].dims);
    EXPECT_EQ("y", model_config->outputs[0].name);
    EXPECT_EQ(neuropod::TensorType::FLOAT_TENSOR, model_config->outputs[0].type);
    EXPECT_EQ(std::vector<neuropod::Dimension>({-1, 2, neuropod::Dimension("some_symbol")}),
              model_config->outputs[0].dims);
}

TEST(test_config_utils, invalid_name)
{
    // Name must be a string
    std::istringstream config(replace("\"addition_model\"", "true"));
    EXPECT_THROW(neuropod::load_model_config(config), std::runtime_error);
}

TEST(test_config_utils, invalid_platform)
{
    // Platform must be a string
    std::istringstream config(replace("\"tensorflow\"", "5"));
    EXPECT_THROW(neuropod::load_model_config(config), std::runtime_error);
}

TEST(test_config_utils, invalid_spec_dtype)
{
    // complex128 is not a supported type
    std::istringstream config(replace("float32", "complex128"));
    EXPECT_THROW(neuropod::load_model_config(config), std::runtime_error);
}

TEST(test_config_utils, invalid_spec_name)
{
    // The name of a tensor must be a string
    std::istringstream config(replace("\"x\"", "true"));
    EXPECT_THROW(neuropod::load_model_config(config), std::runtime_error);
}

TEST(test_config_utils, invalid_spec_shape)
{
    // "123" is not a valid shape. Must be an array
    std::istringstream config(replace("[null, 2, \"some_symbol\"]", "\"123\""));
    EXPECT_THROW(neuropod::load_model_config(config), std::runtime_error);
}

TEST(test_config_utils, invalid_spec_shape_element)
{
    // true is not valid in a shape
    std::istringstream config(replace("[null, 2, \"some_symbol\"]", "[null, 2, \"some_symbol\", true]"));
    EXPECT_THROW(neuropod::load_model_config(config), std::runtime_error);
}

TEST(test_config_utils, invalid_device)
{
    std::istringstream config(replace(R"("x": "GPU")", R"("x": "TPU")"));
    EXPECT_THROW(neuropod::load_model_config(config), std::runtime_error);
}

TEST(test_config_utils, device_cpu)
{
    std::istringstream config(replace(R"("x": "GPU")", R"("x": "CPU")"));
    neuropod::load_model_config(config);
}
