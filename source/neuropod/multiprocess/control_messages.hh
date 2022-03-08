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

#pragma once

#include <ostream>

namespace neuropod
{

// Messages used in the control channel between the main process and the worker
enum MessageType
{
    // Sent by the main process with the neuropod path
    // Valid next messages: LOAD_SUCCESS
    LOAD_NEUROPOD,

    // Sent by the worker process to confirm that the model has been successfully
    // loaded.
    // Valid next messages: ADD_INPUT, LOAD_NEUROPOD
    LOAD_SUCCESS,

    // Sent by the main process when passing tensors to the worker process
    // Valid next messages: INFER
    ADD_INPUT,

    // Sent by the main process once all inputs have been added and we're ready
    // to run inference
    // Valid next messages: RETURN_OUTPUT
    INFER,

    // Sent by the worker process when passing tensors to the main process
    // Valid next messages: ADD_INPUT, LOAD_NEUROPOD
    RETURN_OUTPUT,

    // A message sent by the main process to ask the worker to terminate
    // Note: it is valid to send this message at any time.
    SHUTDOWN,

    // A message sent by the worker process to let the main process know there was an exception
    // Note: it is valid to send this message at any time.
    EXCEPTION,
};

// Used to print out the enum names rather than just a number
std::ostream &operator<<(std::ostream &out, const MessageType value);

} // namespace neuropod
