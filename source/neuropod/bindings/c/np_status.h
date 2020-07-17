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

// Inspired by the TensorFlow C API

#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// Used for returning error messages across the C API boundary
typedef struct NP_Status NP_Status;

// Used for creating and deleting new status messages
NP_Status *NP_NewStatus();
void       NP_DeleteStatus(NP_Status *status);

// Possible status codes
typedef enum NP_Code
{
    NEUROPOD_OK    = 0,
    NEUROPOD_ERROR = 1,
} NP_Code;

// Used for getting details about a status
NP_Code NP_GetCode(const NP_Status *status);

// Get the error message (if any)
const char *NP_Message(const NP_Status *status);

#ifdef __cplusplus
}
#endif
