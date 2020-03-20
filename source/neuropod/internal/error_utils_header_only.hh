//
// Uber, Inc. (c) 2020
//

#pragma once

#include <string>

namespace neuropod
{

namespace detail
{

[[noreturn]] void throw_error_hh(const char *file, int line, const char *function, const std::string &message);
[[noreturn]] void throw_error_hh(
    const char *file, int line, const char *function, const std::string &message, size_t size);
[[noreturn]] void throw_error_hh(
    const char *file, int line, const char *function, const std::string &message, size_t size1, size_t size2);

// An error macro for use in user-facing header files. This is done to avoid
// having the release packages depend on fmt and spdlog being in the user's environment
// See error_utils.hh for more
#define NEUROPOD_ERROR_HH(...) neuropod::detail::throw_error_hh(__FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__);

} // namespace detail

} // namespace neuropod
