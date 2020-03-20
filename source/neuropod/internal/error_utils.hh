//
// Uber, Inc. (c) 2019
//

#pragma once

#include "fmt/format.h"
#include "fmt/ostream.h"
#include "neuropod/internal/logging.hh"

namespace neuropod
{

namespace detail
{

template <typename... Params>
[[noreturn]] void throw_error(const char *file, int line, const char *function, Params &&... params)
{
    // Log the error
    // (this is effectively what the SPDLOG_ERROR macro expands to)
    spdlog::default_logger_raw()->log(
        spdlog::source_loc{file, line, function}, spdlog::level::err, std::forward<Params>(params)...);

    // Throw a runtime error
    throw std::runtime_error("Neuropod Error: " + fmt::format(std::forward<Params>(params)...));
}

} // namespace detail

// A helper macro that lets us do things like
// NEUROPOD_ERROR("Expected value {}, but got {}", a, b);
// This will log the error message and then throw an exception
#define NEUROPOD_ERROR(...) neuropod::detail::throw_error(__FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__);

} // namespace neuropod
