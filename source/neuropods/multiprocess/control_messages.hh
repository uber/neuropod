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

// Validates that state machine transitions are happening correctly
class TransitionVerifier {
private:
    MessageType last_type_;
    bool is_first_message_ = true;

public:
    // Verifies that a state transition is allowed from the last state
    // to the current state
    void assert_transition_allowed(MessageType current_type)
    {
        if (current_type == HEARTBEAT || current_type == SHUTDOWN)
        {
            // These messages are allowed at any time
            return;
        }

        // Special case for the first message
        if (is_first_message_ && current_type != LOAD_NEUROPOD)
        {
            NEUROPOD_ERROR("OPE: Invalid state transition. Expected LOAD_NEUROPOD as first state. Got " << current_type);
        }

        // Using `set` instead of `unordered_set` because it doesn't require the type to be
        // hashable
        static const std::set<std::pair<MessageType, MessageType>> allowed_transitions = {
            std::make_pair(LOAD_NEUROPOD, ADD_INPUT),
            std::make_pair(ADD_INPUT, ADD_INPUT),
            std::make_pair(ADD_INPUT, INFER),
            std::make_pair(INFER, RETURN_OUTPUT),
            std::make_pair(RETURN_OUTPUT, RETURN_OUTPUT),
            std::make_pair(RETURN_OUTPUT, END_OUTPUT),
            std::make_pair(END_OUTPUT, INFER_COMPLETE),
            std::make_pair(INFER_COMPLETE, ADD_INPUT),
            std::make_pair(INFER_COMPLETE, LOAD_NEUROPOD),
        };

        if (!is_first_message_ && allowed_transitions.find(std::make_pair(last_type_, current_type)) == allowed_transitions.end())
        {
            NEUROPOD_ERROR("OPE: Invalid state transition. Got transition from state " << last_type_ << " to " << current_type);
        }

        last_type_ = current_type;
        is_first_message_ = false;
    }
};

// We can batch multiple tensors into a single message in order to minimize
// communication overhead. This is the maximum number of tensors we can include
// in a single message
constexpr int MAX_NUM_TENSORS_PER_MESSAGE = 20;

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
