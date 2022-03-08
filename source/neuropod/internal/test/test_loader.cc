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
#include "neuropod/internal/neuropod_loader.hh"

TEST(test_loader, test_sha)
{
    auto loader = neuropod::get_loader("neuropod/tests/test_data/pytorch_addition_model/");
    EXPECT_EQ(loader->get_hash_for_file("0/data/random_content"),
              "9ac0d09c343ccce2f317fc395d6253f6e3531cc863acbda09e90c7ecdafa5b10");
}
