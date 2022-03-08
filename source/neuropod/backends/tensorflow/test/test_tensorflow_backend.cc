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

#include "neuropod/tests/test_utils.hh"

TEST(test_models, test_tensorflow_addition_model)
{
    // Test the TensorFlow addition model using the native TensorFlow backend
    test_addition_model("neuropod/tests/test_data/tf_addition_model/");
}

TEST(test_models, test_tensorflow_strings_model)
{
    // Test the TensorFlow strings model using the native TensorFlow backend
    test_strings_model("neuropod/tests/test_data/tf_strings_model/");
}

TEST(test_multiprocess_backend, test_tensorflow_addition_model)
{
    // Test the TensorFlow addition model in another process
    test_addition_model_ope("neuropod/tests/test_data/tf_addition_model/");
}

TEST(test_multiprocess_backend, test_tensorflow_strings_model)
{
    // Test the TensorFlow strings model in another process
    test_strings_model_ope("neuropod/tests/test_data/tf_strings_model/");
}