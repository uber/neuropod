//
// Uber, Inc. (c) 2019
//

#pragma once

// Don't disable any logging at compile time
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "spdlog/fmt/ostr.h"
#include "spdlog/spdlog.h"

namespace neuropod
{

// Initialize Neuropod logging and set the initial log level (if not already initialized)
// This is used by code that logs as soon as the process is loaded. Object construction order
// across translation units is not defined so if you attempt to log before the process has
// fully started, the messages may not show up. Calling `init_logging` fixes this
void init_logging();

} // namespace neuropod
