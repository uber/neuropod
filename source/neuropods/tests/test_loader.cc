//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"
#include "neuropods/internal/neuropod_loader.hh"

TEST(test_loader, test_sha)
{
    auto loader = neuropods::get_loader("neuropods/tests/test_data/pytorch_addition_model/");
    EXPECT_EQ(loader->get_hash_for_file("0/data/random_content"),
              "9ac0d09c343ccce2f317fc395d6253f6e3531cc863acbda09e90c7ecdafa5b10");
}
