//
// Uber, Inc. (c) 2019
//

#include "neuropods/internal/logging.hh"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <atomic>
#include <sstream>

namespace neuropods
{

namespace
{
// Get a default log level from the environment
LogSeverity get_default_log_level()
{
    const char * log_level_cstr = std::getenv("NEUROPOD_LOG_LEVEL");
    if (log_level_cstr == nullptr)
    {
        return LogSeverity::INFO;
    }

    std::string log_level(log_level_cstr);
    if (log_level == "DEBUG")
    {
        return LogSeverity::DEBUG;
    }

    if (log_level == "INFO")
    {
        return LogSeverity::INFO;
    }

    if (log_level == "WARN")
    {
        return LogSeverity::WARN;
    }

    if (log_level == "ERROR")
    {
        return LogSeverity::ERROR;
    }

    std::cerr << "Warning: Invalid value for NEUROPOD_LOG_LEVEL: " << log_level << ". Falling back to INFO" << std::endl;
    return LogSeverity::INFO;
}

std::atomic<LogSeverity> current_log_level{get_default_log_level()};

}

void set_log_level(LogSeverity severity)
{
    current_log_level = severity;
}

void log(LogSeverity severity, const char *fname, int line, const std::string &message)
{
    if (severity < current_log_level)
    {
        // Don't log
        return;
    }

    // Based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/default/logging.cc#L96
    const auto now_micros = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    const time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
    const int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
    const size_t time_buffer_size = 30;
    char time_buffer[time_buffer_size];
    strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S",
             localtime(&now_seconds));

    fprintf(stderr, "%s.%06d: %c %s:%d] %s\n", time_buffer, micros_remainder,
            "DIWE"[static_cast<int>(severity)], fname, line, message.c_str());
}

} // namespace neuropods
