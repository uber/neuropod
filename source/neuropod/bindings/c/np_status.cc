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

#include "neuropod/bindings/c/np_status_internal.h"

// Used for creating and deleting new status messages
// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
NP_Status *NP_NewStatus()
{
    return new NP_Status();
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
void NP_DeleteStatus(NP_Status *status)
{
    delete status;
}

// Clear a status
// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
void NP_ClearStatus(NP_Status *status)
{
    status->code = NEUROPOD_OK;
    status->message.clear();
}

// Used for getting details about a status
// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
NP_Code NP_GetCode(const NP_Status *status)
{
    return status->code;
}

// Get the error message (if any)
// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
const char *NP_GetMessage(const NP_Status *status)
{
    return status->message.c_str();
}
