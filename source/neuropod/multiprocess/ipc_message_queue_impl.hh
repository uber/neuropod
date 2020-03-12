//
// Uber, Inc. (c) 2020
//

#pragma once

#include "neuropod/multiprocess/ipc_message_queue.hh"

namespace ipc = boost::interprocess;

namespace neuropod
{

namespace detail
{

// The max size for the send and recv control queues
constexpr auto MAX_QUEUE_SIZE = 20;

enum QueueMessageType
{
    // Contains user defined data. The payload of this type of message is
    // not handled by the queue directly and is added to `out_queue_`
    USER_PAYLOAD,

    // A heartbeat message
    HEARTBEAT,

    // A DONE message sent to signify that the specified message id
    // went out of scope in the sending process (i.e. that the sending process is "done"
    // with that message). This means we can drop our references to any transferrables
    // tied to that message
    DONE,

    // Shutdown the queues
    SHUTDOWN_QUEUES,
};

// Used to generate IDs for messages
extern std::atomic_uint64_t msg_counter;

// The on-the-wire format of the data
// UserPayloadType should be an enum that specifies types of payloads
template <typename UserPayloadType>
struct __attribute__((__packed__)) WireFormat
{
    // The ID of the message
    uint64_t id;

    // The type of the message
    QueueMessageType type;

    // Whether or not this message requires a DONE message
    // to be sent when it goes out of scope in the receiving
    // process
    bool requires_done_msg = false;

    // Whether or not the payload is inline
    bool is_inline;

    // The size of the payload in bytes
    uint32_t payload_size;

    // A user-defined type of the payload
    // Note: this field is only checked if `type` is USER_PAYLOAD
    UserPayloadType payload_type;

    union {
        // An inline payload
        char payload[8192];

        // An SHM id of the actual payload
        // This is used for large messages that are serialized and put in
        // shm
        char payload_id[24];
    };

    WireFormat() = default;

    // Delete the copy constructor and the copy assignment operator
    WireFormat(const WireFormat<UserPayloadType> &) = delete;
    WireFormat &operator=(const WireFormat<UserPayloadType> &other) = delete;

    // Keep the move constructor and move assignment operator
    WireFormat(WireFormat<UserPayloadType> &&) = default;
    WireFormat &operator=(WireFormat<UserPayloadType> &&other) = default;
};

// Serialize a payload into `data` and add any created transferrables to `transferrables`
// If the payload is small enough (less than the size of `payload_` in the wire format), it will be
// stored inline in the message. Otherwise it'll be serialized and put into a shared memory
// block. That block will be added to `transferrables` to ensure it stays in scope while the message
// is in transit.
template <typename Payload, typename UserPayloadType>
void serialize_payload(const Payload &payload, WireFormat<UserPayloadType> &data, Transferrables &transferrables)
{
    // Serialize the payload
    std::stringstream ss;
    ipc_serialize(ss, payload);

    // Set the size
    auto size_bytes   = ss.tellp();
    data.payload_size = size_bytes;

    if (size_bytes <= sizeof(data.payload))
    {
        // We can store this message inline
        ss.read(data.payload, size_bytes);
        data.is_inline = true;
    }
    else
    {
        // Store in SHM and set msg.payload_id
        SPDLOG_DEBUG("Could not fit data in inline message. Sending via SHM. Requested size: {}", size_bytes);
        SHMBlockID block_id;

        auto block = shm_allocator.allocate_shm(size_bytes, block_id);

        // Write the serialized message into the block
        ss.read(static_cast<char *>(block.get()), size_bytes);

        // Copy the block id into the message
        memcpy(data.payload_id, block_id.data(), sizeof(data.payload_id));
        data.is_inline = false;

        // Add this block to our list of transferrables so it stays in scope until
        // the other process reads the message
        transferrables.emplace_back(std::move(block));
    }
}

// Get a payload of type `Payload` from a message
template <typename Payload, typename UserPayloadType>
void deserialize_payload(const WireFormat<UserPayloadType> &data, Payload &out)
{
    std::stringstream ss;
    if (data.is_inline)
    {
        // The message is inline so we can just read it
        ss.write(data.payload, data.payload_size);
    }
    else
    {
        // The message is stored in SHM
        SHMBlockID block_id;
        memcpy(block_id.data(), data.payload_id, sizeof(data.payload_id));

        // Load the block and get the data
        auto block = shm_allocator.load_shm(block_id);
        ss.write(static_cast<char *>(block.get()), data.payload_size);
    }

    ipc_deserialize(ss, out);
}

} // namespace detail

// The worker loop for the message reading thread
template <typename UserPayloadType>
void IPCMessageQueue<UserPayloadType>::read_worker_loop()
{
    while (true)
    {
        // Compute the timeout
        auto timeout_at =
            boost::interprocess::microsec_clock::universal_time() + boost::posix_time::milliseconds(MESSAGE_TIMEOUT_MS);

        // Get a message
        auto         received = stdx::make_unique<WireFormat>();
        size_t       received_size;
        unsigned int priority;
        bool         successful_read =
            recv_queue_->timed_receive(received.get(), sizeof(WireFormat), received_size, priority, timeout_at);

        if (!successful_read)
        {
            // We timed out
            NEUROPOD_ERROR("Timed out waiting for a response from worker process. "
                           "Didn't receive a message in {}ms, but expected a heartbeat every {}ms.",
                           MESSAGE_TIMEOUT_MS,
                           HEARTBEAT_INTERVAL_MS);
        }

        if (received->type != detail::USER_PAYLOAD)
        {
            SPDLOG_TRACE("OPE: Read thread received IPC control message {}.", received->type);
        }

        if (received->type == detail::HEARTBEAT)
        {
            // This is a heartbeat message so continue
            continue;
        }
        else if (received->type == detail::DONE)
        {
            // Handle DONE messages by erasing all the in_transit items for that message
            uint64_t acked_id;
            detail::deserialize_payload(*received, acked_id);

            std::lock_guard<std::mutex> lock(in_transit_mutex_);
            in_transit_.erase(acked_id);
        }
        else if (received->type == detail::SHUTDOWN_QUEUES)
        {
            // Shutdown once we've received DONEs for all the messages we've sent
            shutdown_started_ = true;

            // Note: we're using the `try_` variant to avoid blocking shutdown here
            // Note: out_queue_ should only have one listener at any given time
            // (since recv isn't threadsafe). Because of this, this message should
            // wake up a user thread that is blocked (if any)
            out_queue_.try_emplace(std::move(received));
        }
        else
        {
            // This is a user-handled message
            out_queue_.emplace(std::move(received));
        }

        if (shutdown_started_)
        {
            if (in_transit_.empty())
            {
                // We can finish shutting down
                break;
            }
            else
            {
                SPDLOG_TRACE("OPE: Tried to shut down read worker thread, but still waiting on {} `DONE` messages.",
                             in_transit_.size());
            }
        }
    }
}

// Send a message to the other process
template <typename UserPayloadType>
void IPCMessageQueue<UserPayloadType>::send_message(const WireFormat &msg)
{
    if (msg.type == detail::USER_PAYLOAD)
    {
        SPDLOG_TRACE("OPE: Sending user payload of type: {}", msg.payload_type);
    }
    else
    {
        SPDLOG_TRACE("OPE: Sending IPC control message {}.", msg.type);
    }

    send_queue_->send(&msg, sizeof(msg), 0);
}

// The worker loop for the heartbeat thread
template <typename UserPayloadType>
void IPCMessageQueue<UserPayloadType>::send_heartbeat_loop()
{
    while (send_heartbeat_)
    {
        // Attempt to send a heartbeat message
        WireFormat msg;
        msg.type = detail::HEARTBEAT;
        send_message(msg);

        // Using a condition variable lets us wake up while we're waiting
        std::unique_lock<std::mutex> lk(heartbeat_mutex_);
        heartbeat_cv_.wait_for(
            lk, std::chrono::milliseconds(HEARTBEAT_INTERVAL_MS), [&] { return send_heartbeat_ != true; });
    }
}

template <typename UserPayloadType>
IPCMessageQueue<UserPayloadType>::IPCMessageQueue(const std::string &control_queue_name, ProcessType type)
    : out_queue_(detail::MAX_QUEUE_SIZE),
      control_queue_name_(control_queue_name),
      send_queue_(stdx::make_unique<ipc::message_queue>(ipc::open_or_create,
                                                        ("neuropod_" + control_queue_name_ + "_tw").c_str(),
                                                        detail::MAX_QUEUE_SIZE,
                                                        sizeof(WireFormat))),
      recv_queue_(stdx::make_unique<ipc::message_queue>(ipc::open_or_create,
                                                        ("neuropod_" + control_queue_name_ + "_fw").c_str(),
                                                        detail::MAX_QUEUE_SIZE,
                                                        sizeof(WireFormat))),
      read_worker_(&IPCMessageQueue<UserPayloadType>::read_worker_loop, this),
      heartbeat_thread_(&IPCMessageQueue<UserPayloadType>::send_heartbeat_loop, this)
{
    if (type == WORKER_PROCESS)
    {
        // Switch the send and recv queues
        std::swap(send_queue_, recv_queue_);
    }
}

template <typename UserPayloadType>
IPCMessageQueue<UserPayloadType>::~IPCMessageQueue()
{
    // Join the heartbeat thread
    {
        std::lock_guard<std::mutex> lk(heartbeat_mutex_);
        send_heartbeat_ = false;
    }

    heartbeat_cv_.notify_all();
    heartbeat_thread_.join();

    // Send a shutdown message to ourselves
    WireFormat msg;
    msg.type = detail::SHUTDOWN_QUEUES;
    SPDLOG_TRACE("OPE: Shutting down read thread...");
    recv_queue_->send(&msg, sizeof(msg), 0);

    // Join the read thread
    read_worker_.join();
}

// Send a message with a payload
// Note: this is threadsafe
template <typename UserPayloadType>
template <typename Payload>
void IPCMessageQueue<UserPayloadType>::send_message(UserPayloadType payload_type, const Payload &payload)
{
    // Create a message
    WireFormat msg;
    msg.id           = detail::msg_counter++;
    msg.type         = detail::USER_PAYLOAD;
    msg.payload_type = payload_type;

    // Set the payload
    detail::Transferrables transferrables;
    detail::serialize_payload(payload, msg, transferrables);

    // Check if there are any transferrable items attached
    if (!transferrables.empty())
    {
        // Insert the transferrables into our map of items to store
        std::lock_guard<std::mutex> lock(in_transit_mutex_);
        for (auto &transferrable : transferrables)
        {
            in_transit_.emplace(msg.id, std::move(transferrable));
        }

        msg.requires_done_msg = true;
    }

    // Send the message
    send_message(msg);
}

// Send a message with a payload and ensure `payload` stays in
// scope until the other process is done using the message.
// Note: this is threadsafe
template <typename UserPayloadType>
template <typename Payload>
void IPCMessageQueue<UserPayloadType>::send_message_move(UserPayloadType payload_type, Payload payload)
{
    // Create a message
    WireFormat msg;
    msg.id           = detail::msg_counter++;
    msg.type         = detail::USER_PAYLOAD;
    msg.payload_type = payload_type;

    // Set the payload
    detail::Transferrables transferrables;
    detail::serialize_payload(payload, msg, transferrables);

    // Add the payload to transferrables
    transferrables.emplace_back(std::move(payload));

    // Check if there are any transferrable items attached
    if (!transferrables.empty())
    {
        // Insert the transferrables into our map of items to store
        std::lock_guard<std::mutex> lock(in_transit_mutex_);
        for (auto &transferrable : transferrables)
        {
            in_transit_.emplace(msg.id, std::move(transferrable));
        }

        msg.requires_done_msg = true;
    }

    // Send the message
    send_message(msg);
}

// Send a message with just a payload_type
// Note: this is threadsafe
template <typename UserPayloadType>
void IPCMessageQueue<UserPayloadType>::send_message(UserPayloadType payload_type)
{
    WireFormat msg;
    msg.type         = detail::USER_PAYLOAD;
    msg.payload_type = payload_type;
    send_message(msg);
}

// Get a message. Blocks if the queue is empty.
// Note: this is _NOT_ threadsafe. There should only be one thread calling `recv_message`
// at a time.
template <typename UserPayloadType>
QueueMessage<MessageType> IPCMessageQueue<UserPayloadType>::recv_message()
{
    auto shared_this = this->shared_from_this();

    // Read the message
    std::unique_ptr<WireFormat> out;
    out_queue_.pop(out);
    SPDLOG_TRACE(
        "OPE: Received user payload of type: {} (requires done: {})", out->payload_type, out->requires_done_msg);

    // Convert this to a shared ptr with a deleter that acks the message
    std::shared_ptr<WireFormat> received_shared(out.release(), [shared_this](WireFormat *msg) {
        if (msg->requires_done_msg)
        {
            // Notify the other process that this message is done being read from
            // and any associated resources can be freed
            // This is called in the destructor of `Message` and should not be explicitly called
            detail::Transferrables transferrables;

            // Create a message to ack `msg`
            WireFormat ack_msg;
            ack_msg.type = detail::DONE;
            detail::serialize_payload(msg->id, ack_msg, transferrables);

            if (!transferrables.empty())
            {
                // This must be empty otherwise we'll have an infinite DONE chain
                NEUROPOD_ERROR("[OPE] Transferrables must be empty when sending a `DONE` message.");
            }

            // Send the message
            shared_this->send_message(ack_msg);
        }

        delete msg;
    });

    return QueueMessage<MessageType>(std::move(received_shared));
}

} // namespace neuropod