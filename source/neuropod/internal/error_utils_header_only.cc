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

#include "neuropod/internal/error_utils_header_only.hh"

#include "neuropod/internal/error_utils.hh"

namespace neuropod::detail
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

} // namespace neuropod::detail
