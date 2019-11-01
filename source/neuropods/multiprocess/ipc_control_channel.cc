//
// Uber, Inc. (c) 2019
//

#include "neuropods/multiprocess/ipc_control_channel.hh"

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/internal/logging.hh"
#include "neuropods/multiprocess/control_messages.hh"
#include "neuropods/multiprocess/shm_tensor.hh"

namespace neuropods
{

namespace
{

// The max size for the send and recv control queues
constexpr auto MAX_QUEUE_SIZE = 20;

} // namespace

void TransitionVerifier::assert_transition_allowed(MessageType current_type)
{
    std::lock_guard<std::mutex> lock(mutex_);
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

    if (!is_first_message_ &&
        allowed_transitions.find(std::make_pair(last_type_, current_type)) == allowed_transitions.end())
    {
        NEUROPOD_ERROR("OPE: Invalid state transition. Got transition from state " << last_type_ << " to "
                                                                                   << current_type);
    }

    last_type_        = current_type;
    is_first_message_ = false;
}

IPCControlChannel::IPCControlChannel(const std::string &control_queue_name, ProcessType type)
    : control_queue_name_(control_queue_name),
      send_queue_(stdx::make_unique<ipc::message_queue>(ipc::open_or_create,
                                                        ("neuropod_" + control_queue_name_ + "_tw").c_str(),
                                                        MAX_QUEUE_SIZE,
                                                        sizeof(control_message))),
      recv_queue_(stdx::make_unique<ipc::message_queue>(ipc::open_or_create,
                                                        ("neuropod_" + control_queue_name_ + "_fw").c_str(),
                                                        MAX_QUEUE_SIZE,
                                                        sizeof(control_message)))
{
    if (type == WORKER_PROCESS)
    {
        // Switch the send and recv queues
        std::swap(send_queue_, recv_queue_);
    }
}

IPCControlChannel::~IPCControlChannel() = default;

// Utility to send a message to a message queue
// Boost message_queue is threadsafe so we don't need synchronization here
// assert_transition_allowed is also threadsafe
void IPCControlChannel::send_message(control_message &msg)
{
    // Make sure that it is valid to go from the previous message type to the current one
    verifier_.assert_transition_allowed(msg.type);
    SPDLOG_DEBUG("OPE: Sending message {}", msg.type);
    send_queue_->send(&msg, sizeof(control_message), 0);
}

// Utility to send a message with no content to a message queue
void IPCControlChannel::send_message(MessageType type)
{
    control_message msg;
    msg.type = type;
    send_message(msg);
}

// Utility to send a NeuropodValueMap to a message queue
void IPCControlChannel::send_message(MessageType type, const NeuropodValueMap &data)
{
    control_message msg;
    msg.type        = type;
    msg.num_tensors = 0;

    for (const auto &entry : data)
    {
        const auto &block_id =
            std::dynamic_pointer_cast<NativeDataContainer<SHMBlockID>>(entry.second)->get_native_data();

        // Get the current index
        const auto current_index = msg.num_tensors;

        // Increment the number of tensors
        msg.num_tensors++;

        // Copy in the tensor name
        if (entry.first.length() >= 256)
        {
            NEUROPOD_ERROR("For the multiprocess backend, tensor names must have less than 256 characters. Tried using "
                           "a tensor with name: "
                           << entry.first);
        }

        strncpy(msg.tensor_name[current_index], entry.first.c_str(), 256);

        // Copy in the block ID
        static_assert(std::tuple_size<SHMBlockID>::value == sizeof(msg.tensor_id[0]),
                      "The size of SHMBlockID should match the size of the IDs in control_message");
        memcpy(msg.tensor_id[current_index], block_id.data(), sizeof(msg.tensor_id[current_index]));

        // Send the message if needed
        if (msg.num_tensors == MAX_NUM_TENSORS_PER_MESSAGE)
        {
            send_message(msg);
            msg.num_tensors = 0;
        }
    }

    if (data.size() == 0 || msg.num_tensors != 0)
    {
        // Send the last message
        // (or the only message if we're attempting to send an empty map)
        send_message(msg);
    }
}

// Receive a message
void IPCControlChannel::recv_message(control_message &received)
{
    // Get a message
    size_t       received_size;
    unsigned int priority;
    recv_queue_->receive(&received, sizeof(control_message), received_size, priority);

    SPDLOG_DEBUG("OPE: Received message {}", received.type);

    // Make sure that it is valid to go from the previous message type to the current one
    verifier_.assert_transition_allowed(received.type);
}

// Receive a message with a timeout
bool IPCControlChannel::recv_message(control_message &received, size_t timeout_ms)
{
    // Get a message
    size_t       received_size;
    unsigned int priority;

    auto timeout_at =
        boost::interprocess::microsec_clock::universal_time() + boost::posix_time::milliseconds(timeout_ms);

    bool successful_read =
        recv_queue_->timed_receive(&received, sizeof(control_message), received_size, priority, timeout_at);
    if (successful_read)
    {
        SPDLOG_DEBUG("OPE: Received message {}", received.type);

        // Make sure that it is valid to go from the previous message type to the current one
        verifier_.assert_transition_allowed(received.type);
    }

    return successful_read;
}

void IPCControlChannel::cleanup()
{
    // Delete the control channels
    ipc::message_queue::remove(("neuropod_" + control_queue_name_ + "_tw").c_str());
    ipc::message_queue::remove(("neuropod_" + control_queue_name_ + "_fw").c_str());
}

} // namespace neuropods
