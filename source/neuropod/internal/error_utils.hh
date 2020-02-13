//
// Uber, Inc. (c) 2019
//

#pragma once

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "neuropod/internal/logging.hh"

namespace neuropod
{

// A helper macro that lets us do things like
// NEUROPOD_ERROR("Expected value {}, but got {}", a, b);
// This will log the error message and then throw an exception
#define NEUROPOD_ERROR(...)                                                      \
    do                                                                           \
    {                                                                            \
        SPDLOG_ERROR(__VA_ARGS__);                                               \
        throw std::runtime_error("Neuropod Error: " + fmt::format(__VA_ARGS__)); \
    } while (0);

} // namespace neuropod
