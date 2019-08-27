//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/internal/error_utils.hh"

#include <set>

namespace neuropods
{

// Messages used in the control channel between the main process and the worker
enum MessageType
{
    // Sent by the main process with the neuropod path
    // Valid next messages: ADD_INPUT
    LOAD_NEUROPOD,

    // Sent by the main process when passing tensors to the worker process
    // Valid next messages: ADD_INPUT, INFER
    ADD_INPUT,

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
static_assert(MESSAGE_TIMEOUT_MS > HEARTBEAT_INTERVAL_MS);

// TODO(vip): split into multiple structs
struct __attribute__((__packed__)) control_message
{
    MessageType type;

    // Only used if the message type is ADD_INPUT or RETURN_OUTPUT
    size_t num_tensors;

    // Only used if the message type is ADD_INPUT or RETURN_OUTPUT
    char tensor_id[MAX_NUM_TENSORS_PER_MESSAGE][24];

    // Only used if the message type is ADD_INPUT or RETURN_OUTPUT
    char tensor_name[MAX_NUM_TENSORS_PER_MESSAGE][256];

    // Linux defines the max path length as 4096 (including a NULL char)
    // https://github.com/torvalds/linux/blob/master/include/uapi/linux/limits.h
    char neuropod_path[4096];
};

} // namespace neuropods
