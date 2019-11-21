//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"
#include "neuropods/backends/test_backend/test_neuropod_backend.hh"
#include "neuropods/internal/memory_utils.hh"
#include "neuropods/neuropods.hh"

TEST(test_test_backend, init)
{
    // Make sure we can load a neuropod with garbage data and run inference without crashing
    auto config = neuropods::stdx::make_unique<neuropods::ModelConfig>(
        neuropods::ModelConfig{"name", "platform", {}, {}, {}, {}});
    neuropods::TestNeuropodBackend backend("somepath", config, {});
    neuropods::NeuropodValueMap    inputs;
    backend.infer(inputs);
}
