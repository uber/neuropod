//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"

#include "neuropods/internal/logging.hh"

namespace neuropods
{

TEST(test_logging, no_crash)
{
    // Test that we can set the log level and log without crashing
    set_log_level(neuropods::LogSeverity::DEBUG);
    NEUROPOD_LOG_DEBUG("Test logging");
}

} // namespace neuropods
