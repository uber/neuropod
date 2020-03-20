//
// Uber, Inc. (c) 2020
//

#include "neuropod/internal/error_utils_header_only.hh"

#include "neuropod/internal/error_utils.hh"

namespace neuropod
{

namespace detail
{

void throw_error_hh(const char *file, int line, const char *function, const std::string &message)
{
    throw_error(file, line, function, message);
}

void throw_error_hh(const char *file, int line, const char *function, const std::string &message, size_t size)
{
    throw_error(file, line, function, message, size);
}

void throw_error_hh(
    const char *file, int line, const char *function, const std::string &message, size_t size1, size_t size2)
{
    throw_error(file, line, function, message, size1, size2);
}

} // namespace detail

} // namespace neuropod
