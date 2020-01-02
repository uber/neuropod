//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"
#include "neuropod/backends/test_backend/test_neuropod_backend.hh"
#include "neuropod/internal/memory_utils.hh"
#include "neuropod/neuropod.hh"

TEST(test_test_backend, init)
{
    // Make sure we can load a neuropod with garbage data and run inference without crashing
    auto config =
        neuropod::stdx::make_unique<neuropod::ModelConfig>(neuropod::ModelConfig{"name", "platform", {}, {}, {}, {}});
    neuropod::TestNeuropodBackend backend("somepath", config, {});
    neuropod::NeuropodValueMap    inputs;
    backend.infer(inputs);
}
