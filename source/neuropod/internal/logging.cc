//
// Uber, Inc. (c) 2019
//

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
