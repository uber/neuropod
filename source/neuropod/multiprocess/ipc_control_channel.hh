//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/multiprocess/control_messages.hh"

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

class IPCControlChannel
{
private:
    // Control channels for communicating between the main process and worker process
    std::string                         control_queue_name_;
    std::unique_ptr<ipc::message_queue> send_queue_;
    std::unique_ptr<ipc::message_queue> recv_queue_;

    // Verifies that the state machine is operating as expected
    TransitionVerifier verifier_;

public:
    IPCControlChannel(const std::string &control_queue_name, ProcessType type);
    ~IPCControlChannel();

    // Utility to send a message to a message queue
    // Note: this is threadsafe
    void send_message(control_message &msg);

    // Utility to send a message with no content to a message queue
    // Note: this is threadsafe
    void send_message(MessageType type);

    // Utility to send a NeuropodValueMap to a message queue
    // Note: this is threadsafe
    void send_message(MessageType type, const NeuropodValueMap &data);

    // Receive a message
    void recv_message(control_message &received);

    // Receive a message with a timeout
    bool recv_message(control_message &received, size_t timeout_ms);

    // Cleanup the message queues
    void cleanup();
};

} // namespace neuropod
