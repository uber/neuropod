//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/multiprocess/control_messages.hh"

#include <boost/interprocess/ipc/message_queue.hpp>

#include <mutex>

namespace ipc = boost::interprocess;

namespace neuropod
{

// Validates that state machine transitions are happening correctly
class TransitionVerifier
{
private:
    MessageType last_type_;
    bool        is_first_message_ = true;
    std::mutex  mutex_;

public:
    // Verifies that a state transition is allowed from the last state
    // to the current state
    void assert_transition_allowed(MessageType current_type);
};

// Used when creating an IPCControlChannel
enum ProcessType
{
    WORKER_PROCESS,
    MAIN_PROCESS,
};

// The user facing interface for IPC control messages
class ControlMessage
{
private:
    // Forward declare the internal control_message struct
    struct control_message_internal;

    // A pointer to the underlying struct
    std::unique_ptr<control_message_internal> msg_;

    friend class IPCControlChannel;

public:
    ControlMessage();
    ~ControlMessage();

    // Get the type of the message
    MessageType get_type();

    // Get a vector of tensor names from the message.
    // This adds items to `data` without clearing it
    void get_tensor_names(std::vector<std::string> &data);

    // Get a the map data in the message
    // This adds items to `data` without clearing it
    void get_valuemap(NeuropodValueMap &data);

    // Get a load_config message from this ControlMessage
    void get_load_config(ope_load_config &data);
};

class IPCControlChannel
{
private:
    // Control channels for communicating between the main process and worker process
    std::string                         control_queue_name_;
    std::unique_ptr<ipc::message_queue> send_queue_;
    std::unique_ptr<ipc::message_queue> recv_queue_;

    // Verifies that the state machine is operating as expected
    TransitionVerifier verifier_;

    // Alias used in the implementation
    using control_message = ControlMessage::control_message_internal;

    // Utility to send a message to a message queue
    // Note: this is threadsafe
    void send_message(control_message &msg);

    // Utility to send a message to a message queue
    // Does not block if the message queue is full
    // This must be used when sending messages to the main process outside of the
    // normal inference process (e.g. HEARTBEAT messages)
    // Note: this is threadsafe
    bool try_send_message(control_message &msg);

public:
    IPCControlChannel(const std::string &control_queue_name, ProcessType type);
    ~IPCControlChannel();

    // Utility to send a message with no content to a message queue
    // Note: this is threadsafe
    void send_message(MessageType type);

    // Utility to send a message to a message queue
    // Does not block if the message queue is full
    // This must be used when sending messages to the main process outside of the
    // normal inference process (e.g. HEARTBEAT messages)
    // Note: this is threadsafe
    bool try_send_message(MessageType type);

    // Utility to send a vector of tensor names to a message queue
    // Note: this is threadsafe
    void send_message(MessageType type, const std::vector<std::string> &data);

    // Utility to send a NeuropodValueMap to a message queue
    // Note: this is threadsafe
    void send_message(MessageType type, const NeuropodValueMap &data);

    // Utility to send a `ope_load_config` struct to a message queue
    void send_message(MessageType type, const ope_load_config &data);

    // Receive a message
    void recv_message(ControlMessage &received);

    // Receive a message with a timeout
    bool recv_message(ControlMessage &received, size_t timeout_ms);

    // Cleanup the message queues
    void cleanup();
};

} // namespace neuropod
