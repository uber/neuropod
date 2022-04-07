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

#include "neuropod/backends/torchscript/torch_tensor.hh"
#include "neuropod/tests/test_utils.hh"

#include <string>
#include <vector>

#include <stdlib.h>

namespace
{
#if CAFFE2_NIGHTLY_VERSION >= 20220127
void test_torchscript_dict_with_union_value_type_model(neuropod::Neuropod &neuropod)
{
    // Tests a model that has Dict[str, Union[List[str], torch.Tensor]] as model input type
    std::vector<int64_t>           str_shape             = {3};
    std::vector<int64_t>           num_shape             = {3, 1};
    const std::vector<int64_t>     expected_output_shape = {3, 2};
    const std::vector<std::string> a_data                = {"a", "b", "c"};
    const float                    b_data[]              = {1, 2, 3};
    const float                    target[]              = {1, 1, 2, 2, 0, 3};

    auto a_ten = neuropod.allocate_tensor<std::string>(str_shape);
    a_ten->copy_from(a_data);
    auto b_ten = neuropod.allocate_tensor<float>(num_shape);
    b_ten->copy_from(b_data, 3);

    neuropod::NeuropodValueMap input_data;
    input_data["a"] = a_ten;
    input_data["b"] = b_ten;

    const auto output_data = neuropod.infer(input_data);

    const std::vector<float> output_vec = output_data->at("c")->as_typed_tensor<float>()->get_data_as_vector();

    const std::vector<int64_t> out_shape = output_data->at("c")->as_tensor()->get_dims();

    EXPECT_EQ(output_vec.size(), 6);
    EXPECT_TRUE(std::equal(output_vec.begin(), output_vec.end(), target));
    EXPECT_TRUE(out_shape == expected_output_shape);
}
#endif
} // namespace

TEST(test_torchscript_backend, test_torchscript_addition_model)
{
    // Test the TorchScript addition model using the native torchscript backend
    test_addition_model("neuropod/tests/test_data/torchscript_addition_model/");
}

TEST(test_torchscript_backend, test_torchscript_addition_tensor_output_model)
{
    // Test the TorchScript addition model using the native torchscript backend
    test_addition_model("neuropod/tests/test_data/torchscript_addition_model_single_output/");
}

TEST(test_torchscript_backend, test_torchscript_strings_model)
{
    // Test the TorchScript strings model using the native torchscript backend
    test_strings_model("neuropod/tests/test_data/torchscript_strings_model/");
}

TEST(test_torchscript_backend, test_torchscript_dict_with_union_value_type_model)
{
#if CAFFE2_NIGHTLY_VERSION >= 20220127
    // Load the neuropod
    neuropod::RuntimeOptions opts;
    opts.load_model_at_construction = false;
    neuropod::Neuropod neuropod("neuropod/tests/test_data/torchscript_dict_with_union_value_type_model/", opts);

    // Should fail because we haven't loaded the model yet
    EXPECT_ANY_THROW(test_torchscript_dict_with_union_value_type_model(neuropod));

    // Load the model and try again
    neuropod.load_model();
    test_torchscript_dict_with_union_value_type_model(neuropod);
#endif
}

TEST(test_torchscript_backend, invalid_dtype)
{
    neuropod::Neuropod model("neuropod/tests/test_data/torchscript_strings_model/");

    // These should work
    model.allocate_tensor<float>({2});
    model.allocate_tensor<uint8_t>({2});

    // These types aren't supported by torch
    EXPECT_THROW(model.allocate_tensor<uint16_t>({2}), std::runtime_error);
    EXPECT_THROW(model.allocate_tensor<uint32_t>({2}), std::runtime_error);
    EXPECT_THROW(model.allocate_tensor<uint64_t>({2}), std::runtime_error);
}

// TODO(vip): reenable this test once we have more complete support for directly loading models
// TEST(test_torchscript_backend, load_model_from_path)
// {
//     // Load a TorchScript model directly
//     neuropod::TorchNeuropodBackend model("neuropod/tests/test_data/torchscript_strings_model/0/data/model.pt");
// }

TEST(test_multiprocess_backend, test_torchscript_addition_model)
{
    // Test the TorchScript addition model in another process
    test_addition_model_ope("neuropod/tests/test_data/torchscript_addition_model/");
}

TEST(test_multiprocess_backend, test_torchscript_strings_model)
{
    // Test the TorchScript strings model in another process
    test_strings_model_ope("neuropod/tests/test_data/torchscript_strings_model/");
}

TEST(test_multiprocess_backend, test_torchscript_dict_with_union_value_type_model)
{
#if CAFFE2_NIGHTLY_VERSION >= 20220127
    neuropod::RuntimeOptions opts;
    opts.use_ope = true;
    neuropod::Neuropod neuropod("neuropod/tests/test_data/torchscript_dict_with_union_value_type_model/", opts);
    test_torchscript_dict_with_union_value_type_model(neuropod);
#endif
}