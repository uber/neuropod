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
#define NEUROPOD_ERROR(...) neuropod::detail::throw_error(__FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__)

} // namespace neuropod
