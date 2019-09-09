//
// Uber, Inc. (c) 2019
//

#pragma once

#include <iostream>
#include <sstream>

namespace neuropods
{

enum LogSeverity
{
    DEBUG,
    INFO,
    WARN,
    ERROR,
};

// Set the log level. All messages >= this level will be logged
void set_log_level(LogSeverity severity);

// Don't call this directly. Use one of the macros below
void log(LogSeverity severity, const char *fname, int line, const std::string &message);

#define NEUROPOD_LOG_WITH_LINE_AND_FILE(severity, fname, line, MSG) \
    {                                                               \
        std::stringstream neuropod_log_msg;                         \
        neuropod_log_msg << MSG;                                    \
        log(severity, fname, line, neuropod_log_msg.str());         \
    }

#define NEUROPOD_LOG(severity, MSG) NEUROPOD_LOG_WITH_LINE_AND_FILE(severity, __FILE__, __LINE__, MSG)


#define NEUROPOD_LOG_DEBUG(MSG) NEUROPOD_LOG(LogSeverity::DEBUG, MSG)
#define NEUROPOD_LOG_INFO(MSG) NEUROPOD_LOG(LogSeverity::INFO, MSG)
#define NEUROPOD_LOG_WARN(MSG) NEUROPOD_LOG(LogSeverity::WARN, MSG)
#define NEUROPOD_LOG_ERROR(MSG) NEUROPOD_LOG(LogSeverity::ERROR, MSG)

} // namespace neuropods
