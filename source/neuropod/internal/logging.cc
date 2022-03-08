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

#include "neuropod/internal/logging.hh"

#include <iostream>

namespace neuropod
{

namespace
{
// Get a default log level from the environment
spdlog::level::level_enum get_default_log_level()
{
    const char *log_level_cstr = std::getenv("NEUROPOD_LOG_LEVEL");
    if (log_level_cstr == nullptr)
    {
        return spdlog::level::info;
    }

    std::string log_level(log_level_cstr);
    if (log_level == "TRACE")
    {
        return spdlog::level::trace;
    }

    if (log_level == "DEBUG")
    {
        return spdlog::level::debug;
    }

    if (log_level == "INFO")
    {
        return spdlog::level::info;
    }

    if (log_level == "WARN")
    {
        return spdlog::level::warn;
    }

    if (log_level == "ERROR")
    {
        return spdlog::level::err;
    }

    std::cerr << "Warning: Invalid value for NEUROPOD_LOG_LEVEL: " << log_level << ". Falling back to INFO"
              << std::endl;
    return spdlog::level::info;
}

std::once_flag logging_initialized;

} // namespace

void init_logging()
{
    std::call_once(logging_initialized, []() {
        spdlog::set_level(get_default_log_level());
        spdlog::set_pattern("%D %T.%f: %L %@] [thread %t, process %P] %v");
    });
}

namespace
{

bool static_init_logging()
{
    init_logging();
    return true;
}

// Initialize logging if nothing else did
bool current_log_level = static_init_logging();

} // namespace

} // namespace neuropod
