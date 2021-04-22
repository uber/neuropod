/* Copyright (c) 2020 UATC, LLC

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

#include <thread>

#include <patchlevel.h>

TEST(test_models, test_pytorch_addition_model)
{
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 8
    // Test the PyTorch addition model using the python bridge
    test_addition_model("neuropod/tests/test_data/pytorch_addition_model_gpu/");
#endif
}

TEST(test_models, test_pytorch_addition_model_threaded)
{
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 8
    std::thread t([]() {
        // Test the PyTorch addition model using the python bridge
        test_addition_model("neuropod/tests/test_data/pytorch_addition_model_gpu/");
    });

    t.join();
#endif
}
