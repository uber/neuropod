//
// Uber, Inc. (c) 2019
//

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
