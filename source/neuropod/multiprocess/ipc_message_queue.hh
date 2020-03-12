//
// Uber, Inc. (c) 2020
//

#pragma once

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/internal/blocking_spsc_queue.hh"
#include "neuropod/multiprocess/ipc_serialization.hh"
#include "neuropod/multiprocess/shm_tensor.hh"

#include <boost/any.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <unordered_map>

namespace ipc = boost::interprocess;

namespace neuropod
{

namespace detail
{

// `Transferrables` are items that must be kept in scope while a message is in transit
// Usually, these contain data that is stored in shared memory and is
// used to ensure the sending process maintains a reference to the
// data until the receiving process is done loading it.
// This is used to implement cross-process "moves" of data in shared memory
using Transferrables = std::vector<boost::any>;

// The on-the-wire format of the data
// UserPayloadType should be an enum that specifies types of payloads
template <typename UserPayloadType>
struct WireFormat;

// Serialize a payload into `data` and add any created transferrables to `transferrables`
// If the payload is small enough (less than the size of `payload_` in the wire format), it will be
// stored inline in the message. Otherwise it'll be serialized and put into a shared memory
// block. That block will be added to `transferrables` to ensure it stays in scope while the message
// is in transit.
template <typename Payload, typename UserPayloadType>
void serialize_payload(const Payload &payload, WireFormat<UserPayloadType> &data, Transferrables &transferrables);

// Get a payload of type `Payload` from a message
template <typename Payload, typename UserPayloadType>
void deserialize_payload(const WireFormat<UserPayloadType> &data, Payload &out);

} // namespace detail

// Forward declare IPCMessageQueue
template <typename>
class IPCMessageQueue;

// This is the user-facing queue message type
// UserPayloadType should be an enum that specifies types of payloads
template <typename UserPayloadType>
class QueueMessage
{
private:
    // A pointer to the underlying data
    std::shared_ptr<detail::WireFormat<UserPayloadType>> data_;

    // Note: The constructor is private and only used in `IPCMessageQueue`
    template <typename>
    friend class IPCMessageQueue;

    // Constructor used when receiving messages
    QueueMessage(std::shared_ptr<detail::WireFormat<UserPayloadType>> data) : data_(std::move(data)) {}

public:
    ~QueueMessage() = default;

    // Get a payload of type `Payload` from this message
    template <typename Payload>
    void get(Payload &out)
    {
        detail::deserialize_payload(*data_, out);
    }

    // Get the type of the user-defined payload included in this message
    // This should only be used when it is known that this message contains
    // a user-defined payload
    UserPayloadType get_payload_type() { return data_->payload_type; }
};

// The type of the process creating the message queue
enum ProcessType
{
    WORKER_PROCESS,
    MAIN_PROCESS,
};

// A bidirectional IPC message queue that supports cross-process moves or copies of payloads.
// Includes an implementation of heartbeats and message acknowledgement (in the form of DONE messages)
// This class starts a thread for reading from the underlying `recv_queue` and starts a thread
// for sending heartbeats.
template <typename UserPayloadType>
class IPCMessageQueue : public std::enable_shared_from_this<IPCMessageQueue<UserPayloadType>>
{
private:
    using WireFormat = detail::WireFormat<UserPayloadType>;

    // A queue to store output messages received by the read thread
    BlockingSPSCQueue<std::unique_ptr<WireFormat>> out_queue_;

    // Internal IPC queues to communicate with the other process
    std::string                         control_queue_name_;
    std::unique_ptr<ipc::message_queue> send_queue_;
    std::unique_ptr<ipc::message_queue> recv_queue_;

    // Stores items in transit
    // A mapping from message_id to items that need to be kept in scope
    // until a DONE message is received for that message id
    // This is used to implement cross-process "moves" of items in
    // shared memory
    std::unordered_multimap<uint64_t, boost::any> in_transit_;
    std::mutex                                    in_transit_mutex_;

    // State needed for the heartbeat thread
    std::atomic_bool        send_heartbeat_{true};
    std::condition_variable heartbeat_cv_;
    std::mutex              heartbeat_mutex_;

    // Whether or not a shutdown is in progress
    bool shutdown_started_ = false;

    // A thread that handles incoming messages
    std::thread read_worker_;

    // A thread that sends heartbeats
    std::thread heartbeat_thread_;

    // The worker loop for the message reading thread
    void read_worker_loop();

    // Send a message to the other process
    void send_message(const WireFormat &msg);

    // The worker loop for the heartbeat thread
    void send_heartbeat_loop();

public:
    IPCMessageQueue(const std::string &control_queue_name, ProcessType type);

    ~IPCMessageQueue();

    // Send a message with a payload
    // Note: this is threadsafe
    template <typename Payload>
    void send_message(UserPayloadType payload_type, const Payload &payload);

    // Send a message with a payload and ensure `payload` stays in
    // scope until the other process is done using the message.
    // Note: this is threadsafe
    template <typename Payload>
    void send_message_move(UserPayloadType payload_type, Payload payload);

    // Send a message with just a payload_type
    // Note: this is threadsafe
    void send_message(UserPayloadType payload_type);

    // Get a message. Blocks if the queue is empty.
    // Note: this is _NOT_ threadsafe. There should only be one thread calling `recv_message`
    // at a time.
    QueueMessage<MessageType> recv_message();
};

// Cleanup control channels for the queue with name `control_queue_name`
void cleanup_control_channels(const std::string &control_queue_name);

} // namespace neuropod

#include "neuropod/multiprocess/ipc_message_queue_impl.hh"