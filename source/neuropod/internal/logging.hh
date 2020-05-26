/* Copyright (c) 2020 UATC, LLC

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

// Don't disable any logging at compile time
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "spdlog/fmt/ostr.h"
#include "spdlog/spdlog.h"

namespace neuropod
{

// Initialize Neuropod logging and set the initial log level (if not already initialized)
// This is used by code that logs as soon as the process is loaded. Object construction order
// across translation units is not defined so if you attempt to log before the process has
// fully started, the messages may not show up. Calling `init_logging` fixes this
void init_logging();

} // namespace neuropod
