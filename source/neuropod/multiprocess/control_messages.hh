//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropod/internal/error_utils.hh"

#include <set>

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
    // Valid next messages: ADD_INPUT
    LOAD_SUCCESS,

    // Sent by the main process when passing tensors to the worker process
    // Valid next messages: ADD_INPUT, REQUEST_OUTPUT, INFER
    ADD_INPUT,

    // Sent by the main process to the worker process to request a subset
    // of the outputs
    // Valid next messages: REQUEST_OUTPUT, INFER
    REQUEST_OUTPUT,

    // Sent by the main process once all inputs have been added and we're ready
    // to run inference
    // Valid next messages: RETURN_OUTPUT
    INFER,

    // Sent by the worker process when passing tensors to the main process
    // Valid next messages: RETURN_OUTPUT, END_OUTPUT
    RETURN_OUTPUT,

    // Sent by the worker process once inference is completed and all outputs
    // have been sent to the main process
    // Valid next messages: INFER_COMPLETE
    END_OUTPUT,

    // Sent by the main process once inference is complete
    // This is used to notify the worker process that it no longer needs to store
    // the model outputs.
    // Valid next messages: ADD_INPUT, LOAD_NEUROPOD
    INFER_COMPLETE,

    // A noop message sent by the worker process used to ensure that the worker
    // is alive
    // Note: it is valid to send this message at any time.
    HEARTBEAT,

    // A message sent by the main process to ask the worker to terminate
    // Note: it is valid to send this message at any time.
    SHUTDOWN,
};

// Used to print out the enum names rather than just a number
std::ostream &operator<<(std::ostream &out, const MessageType value);

// We can batch multiple tensors into a single message in order to minimize
// communication overhead. This is the maximum number of tensors we can include
// in a single message
constexpr int MAX_NUM_TENSORS_PER_MESSAGE = 20;

// The worker process should send a heartbeat every 2 seconds
constexpr int HEARTBEAT_INTERVAL_MS = 2000;

// 5 second timeout for the main process to receive a message from the worker
// This is generous since the worker sends heartbeats every 2 seconds (defined above)
constexpr int MESSAGE_TIMEOUT_MS = 5000;

// Ensure the timeout is larger than the heartbeat interval
static_assert(MESSAGE_TIMEOUT_MS > HEARTBEAT_INTERVAL_MS, "Message timeout must be larger than the heartbeat interval");

} // namespace neuropod
