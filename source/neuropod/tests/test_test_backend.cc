//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"
#include "neuropod/backends/test_backend/test_neuropod_backend.hh"
#include "neuropod/internal/memory_utils.hh"
#include "neuropod/neuropod.hh"

TEST(test_test_backend, init)
{
    // Make sure we can create a TestNeuropodBackend and run inference without crashing
    neuropod::TestNeuropodBackend backend("somepath", {});
    neuropod::NeuropodValueMap    inputs;
    backend.infer(inputs);
}
