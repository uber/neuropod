//
// Uber, Inc. (c) 2020
//

#pragma once

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/multiprocess/ipc_serialization.hh"
#include "neuropod/multiprocess/shm_tensor.hh"

#include <boost/any.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <queue>
#include <thread>
#include <unordered_map>

namespace ipc = boost::interprocess;

namespace neuropod
{

namespace detail
{

// The max size for the send and recv control queues
constexpr auto MAX_QUEUE_SIZE = 20;

// Items that must be kept in scope while a message is in transit
// This is used to implement cross-process moves
using Transferrables = std::vector<boost::any>;

enum QueueMessageType
{
    // Contains user defined data. The payload of this type of message is
    // not handled by the queue directly
    USER_PAYLOAD,

    // A heartbeat message
    HEARTBEAT,

    // An ACK message. Used for cross process moves of transferrables
    ACK,

    // Shutdown the queues
    SHUTDOWN_QUEUES,
};

// Used to generate IDs for messages
extern std::atomic_uint64_t msg_counter;

} // namespace detail

// A bounded blocking spsc queue
template <typename T>
class BlockingSPSCQueue
{
private:
    std::queue<T>           queue_;
    std::condition_variable full_cv_;
    std::condition_variable empty_cv_;
    std::mutex              mutex_;

    size_t capacity_;

public:
    BlockingSPSCQueue(size_t capacity) : capacity_(capacity) {}
    ~BlockingSPSCQueue() = default;

    bool try_emplace(T &&item)
    {
        bool success = false;

        // Lock the mutex and add to the queue if we have capacity
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (queue_.size() < capacity_)
            {
                queue_.emplace(std::forward<T>(item));
                success = true;
            }
        }

        // Notify any waiting read threads if we need to
        if (success)
        {
            empty_cv_.notify_all();
        }

        return success;
    }

    void emplace(T &&item)
    {
        // Lock the mutex and add to the queue (or wait until we have capacity)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (queue_.size() >= capacity_)
            {
                full_cv_.wait(lock, [&] { return queue_.size() < capacity_; });
            }

            queue_.emplace(std::forward<T>(item));
        }

        // Notify any waiting read threads
        empty_cv_.notify_all();
    }

    void pop(T &item)
    {
        // Lock the mutex and get an item from the queue (or wait until we have an item)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (queue_.empty())
            {
                empty_cv_.wait(lock, [&] { return !queue_.empty(); });
            }

            item = std::move(queue_.front());
            queue_.pop();
        }

        // Notify any waiting write threads
        full_cv_.notify_all();
    }
};

// class RAIITimer
// {
// private:
//     std::string name_;
//     std::chrono::time_point<std::chrono::steady_clock> start_;

// public:
//     RAIITimer(std::string name) : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}
//     ~RAIITimer()
//     {
//         auto end = std::chrono::steady_clock::now();
//         SPDLOG_TRACE("Timer: {} took {}ns to execute", name_, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count());
//     }
// };

// Forward declare IPCMessageQueue
template <typename>
class IPCMessageQueue;

// The on-the-wire format of the data
// UserPayloadType should be an enum that specifies types of payloads
template <typename UserPayloadType>
class __attribute__((__packed__)) QueueMessage
{
private:
    // The ID of the message
    uint64_t id_;

    // The type of the message
    detail::QueueMessageType type_;

    // Whether or not this message requires an ack
    bool requires_ack_ = false;

    // Whether or not the payload is inline
    bool is_inline_;

    // The size of the payload in bytes
    uint32_t payload_size_;

    // A user-defined type of the payload
    // Note: this field is only checked if `type` is USER_PAYLOAD
    UserPayloadType payload_type_;

    union {
        // An inline payload
        char payload_[8192];

        // An SHM id of the actual payload
        // This is used for large messages that are serialized and put in
        // shm
        char payload_id_[24];
    };

    // Note: all the constructors are private and are only used in `IPCMessageQueue`
    template <typename>
    friend class IPCMessageQueue;

    // Create a message without a payload
    // This is usually used for sending things like HEARTBEAT messages
    QueueMessage(detail::QueueMessageType type)
        : id_(detail::msg_counter++), // Atomically increment and generate an ID for the message
          type_(type)
    {
    }

    // Create a message with a payload
    // If the payload is small enough (less than the size of `payload_` above), it will be
    // stored inline in the message. Otherwise it'll be serialized and put into a shared memory
    // block. That block will be added to `transferrables` to ensure it stays in scope while the message
    // is in transit.
    template <typename Payload>
    QueueMessage(const Payload &payload, detail::Transferrables &transferrables)
        : id_(detail::msg_counter++) // Atomically increment and generate an ID for the message
    {
        // RAIITimer timer("OPE Payload Serialization");

        // Serialize the payload
        std::stringstream ss;
        ipc_serialize(ss, payload);

        // Set the size
        auto size_bytes = ss.tellp();
        payload_size_   = size_bytes;

        if (size_bytes <= sizeof(payload_))
        {
            // We can store this message inline
            ss.read(payload_, size_bytes);
            is_inline_ = true;
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
            memcpy(payload_id_, block_id.data(), sizeof(payload_id_));
            is_inline_ = false;

            // Add this block to our list of transferrables so it stays in scope until
            // the other process reads the message
            transferrables.emplace_back(std::move(block));
        }
    }

    // A default constructor used when receiving messages
    QueueMessage() {}

    // TODO(vip): we can't store this here because we don't want this in
    // shared memory
    std::shared_ptr<IPCMessageQueue<UserPayloadType>> ack_queue_;

public:

    // Delete the copy constructor and the copy assignment operator
    QueueMessage(const QueueMessage&) = delete;
    QueueMessage& operator=(const QueueMessage& other) = delete;

    // Keep the move constructor and move assignment operator
    QueueMessage(QueueMessage&&) = default;
    QueueMessage& operator=(QueueMessage&& other) = default;

    ~QueueMessage()
    {
        // Send an ack message if we need to
        if (ack_queue_)
        {
            ack_queue_->done(*this);
        }
    }

    // Get a payload of type `Payload` from this message
    template <typename Payload>
    void get(Payload &out)
    {
        // RAIITimer timer("OPE Payload Deserialization");

        std::stringstream ss;
        if (is_inline_)
        {
            // The message is inline so we can just read it
            ss.write(payload_, payload_size_);
        }
        else
        {
            // The message is stored in SHM
            SHMBlockID block_id;
            memcpy(block_id.data(), payload_id_, sizeof(payload_id_));

            // Load the block and get the data
            auto block = shm_allocator.load_shm(block_id);
            ss.write(static_cast<char *>(block.get()), payload_size_);
        }

        ipc_deserialize(ss, out);
    }

    // Get the type of the user-defined payload included in this message
    // This should only be used when it is known that this message contains
    // a user-defined payload
    UserPayloadType get_payload_type() { return payload_type_; }
};

// The type of the process creating the message queue
enum ProcessType
{
    WORKER_PROCESS,
    MAIN_PROCESS,
};

// A bidirectional IPC message queue that supports cross-process moves or copies of payloads.
// Includes an implementation of heartbeats and message acknowledgement
template <typename UserPayloadType>
class IPCMessageQueue : public std::enable_shared_from_this<IPCMessageQueue<UserPayloadType>>
{
private:
    using Message = QueueMessage<UserPayloadType>;

    // Output messages
    BlockingSPSCQueue<Message>     out_queue_;

    // Internal IPC queues
    std::string                         control_queue_name_;
    std::unique_ptr<ipc::message_queue> send_queue_;
    std::unique_ptr<ipc::message_queue> recv_queue_;

    // Stores items in transit
    // A mapping from message_id to items that need to be kept in scope
    // until an ack is received
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
    std::thread heartbeat_thread_;

    // The worker loop for the message reading thread
    void read_worker_loop()
    {
        while (true)
        {
            // Compute the timeout
            auto timeout_at = boost::interprocess::microsec_clock::universal_time() +
                              boost::posix_time::milliseconds(MESSAGE_TIMEOUT_MS);

            // Get a message
            Message      received;
            size_t       received_size;
            unsigned int priority;
            bool         successful_read =
                recv_queue_->timed_receive(&received, sizeof(Message), received_size, priority, timeout_at);

            if (!successful_read)
            {
                // We timed out
                NEUROPOD_ERROR("Timed out waiting for a response from worker process. "
                               "Didn't receive a message in {}ms, but expected a heartbeat every {}ms.",
                               MESSAGE_TIMEOUT_MS,
                               HEARTBEAT_INTERVAL_MS);
            }

            if (received.type_ != detail::USER_PAYLOAD)
            {
                SPDLOG_TRACE("OPE: Read thread received IPC control message {}.", received.type_);
            }

            if (received.type_ == detail::HEARTBEAT)
            {
                // This is a heartbeat message so continue
                continue;
            }
            else if (received.type_ == detail::ACK)
            {
                // Handle ack messages by erasing all the in_transit items for that message
                uint64_t acked_id;
                received.get(acked_id);

                std::lock_guard<std::mutex> lock(in_transit_mutex_);
                in_transit_.erase(acked_id);
            }
            else if (received.type_ == detail::SHUTDOWN_QUEUES)
            {
                // Shutdown once we've received ACKs for all the messages we've sent
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

            if (shutdown_started_ && in_transit_.empty())
            {
                // We can finish shutting down
                break;
            }
        }
    }

    void send_message(const Message &msg)
    {
        if (msg.type_ == detail::USER_PAYLOAD)
        {
            SPDLOG_TRACE("OPE: Sending user payload of type: {}", msg.payload_type_);
        }
        else
        {
            SPDLOG_TRACE("OPE: Sending IPC control message {}.", msg.type_);
        }

        send_queue_->send(&msg, sizeof(Message), 0);
    }

    // The worker loop for the heartbeat thread
    void send_heartbeat_loop()
    {
        while (send_heartbeat_)
        {
            // Attempt to send a heartbeat message
            Message msg(detail::HEARTBEAT);
            send_message(msg);

            // Using a condition variable lets us wake up while we're waiting
            std::unique_lock<std::mutex> lk(heartbeat_mutex_);
            heartbeat_cv_.wait_for(
                lk, std::chrono::milliseconds(HEARTBEAT_INTERVAL_MS), [&] { return send_heartbeat_ != true; });
        }
    }

public:
    IPCMessageQueue(const std::string &control_queue_name, ProcessType type)
        : out_queue_(detail::MAX_QUEUE_SIZE),
          control_queue_name_(control_queue_name),
          send_queue_(stdx::make_unique<ipc::message_queue>(ipc::open_or_create,
                                                            ("neuropod_" + control_queue_name_ + "_tw").c_str(),
                                                            detail::MAX_QUEUE_SIZE,
                                                            sizeof(Message))),
          recv_queue_(stdx::make_unique<ipc::message_queue>(ipc::open_or_create,
                                                            ("neuropod_" + control_queue_name_ + "_fw").c_str(),
                                                            detail::MAX_QUEUE_SIZE,
                                                            sizeof(Message))),
          read_worker_(&IPCMessageQueue<UserPayloadType>::read_worker_loop, this),
          heartbeat_thread_(&IPCMessageQueue<UserPayloadType>::send_heartbeat_loop, this)
    {
        if (type == WORKER_PROCESS)
        {
            // Switch the send and recv queues
            std::swap(send_queue_, recv_queue_);
        }
    }

    ~IPCMessageQueue()
    {
        // Join the heartbeat thread
        {
            std::lock_guard<std::mutex> lk(heartbeat_mutex_);
            send_heartbeat_ = false;
        }

        heartbeat_cv_.notify_all();
        heartbeat_thread_.join();

        // Send a shutdown message to ourselves
        Message msg(detail::SHUTDOWN_QUEUES);
        SPDLOG_TRACE("OPE: Shutting down read thread...");
        recv_queue_->send(&msg, sizeof(Message), 0);

        // Join the read thread
        read_worker_.join();
    }

    // Send a message with a payload
    // Note: this is threadsafe
    template <typename Payload>
    void send_message(UserPayloadType payload_type, const Payload &payload)
    {
        detail::Transferrables transferrables;
        Message                msg(payload, transferrables);
        msg.type_         = detail::USER_PAYLOAD;
        msg.payload_type_ = payload_type;

        // Check if there are any transferrable items attached
        if (!transferrables.empty())
        {
            // Insert the transferrables into our map of items to store
            std::lock_guard<std::mutex> lock(in_transit_mutex_);
            for (auto &transferrable : transferrables)
            {
                in_transit_.emplace(msg.id_, std::move(transferrable));
            }

            msg.requires_ack_ = true;
        }

        // Send the message
        send_message(msg);
    }

    // Send a message with a payload and ensure `payload` stays in
    // scope until the other process is done using the message.
    // Note: this is threadsafe
    template <typename Payload>
    void send_message_move(UserPayloadType payload_type, Payload payload)
    {
        detail::Transferrables transferrables;
        Message                msg(payload, transferrables);
        msg.type_         = detail::USER_PAYLOAD;
        msg.payload_type_ = payload_type;

        transferrables.emplace_back(std::move(payload));

        // Check if there are any transferrable items attached
        if (!transferrables.empty())
        {
            // Insert the transferrables into our map of items to store
            std::lock_guard<std::mutex> lock(in_transit_mutex_);
            for (auto &transferrable : transferrables)
            {
                in_transit_.emplace(msg.id_, std::move(transferrable));
            }

            msg.requires_ack_ = true;
        }

        // Send the message
        send_message(msg);
    }

    // Send a message with just a payload_type
    // Note: this is threadsafe
    void send_message(UserPayloadType payload_type)
    {
        Message msg(detail::USER_PAYLOAD);
        msg.payload_type_ = payload_type;
        send_message(msg);
    }

    // Get a message. Blocks if the queue is empty.
    // Note: this is _NOT_ threadsafe. There should only be one thread calling `recv_message`
    // at a time.
    Message recv_message()
    {
        Message out;
        out_queue_.pop(out);

        SPDLOG_TRACE("OPE: Received user payload of type: {} (requires ack: {})", out.payload_type_, out.requires_ack_);

        if (out.requires_ack_)
        {
            out.ack_queue_ = this->shared_from_this();
        }
        else
        {
            out.ack_queue_ = nullptr;
        }

        return out;
    }

    // Notify the other process that this message is done being read from
    // and any associated resources can be freed
    // This is called in the destructor of `Message` and should not be explicitly called
    void done(const Message &msg)
    {
        detail::Transferrables transferrables;

        // Create a message to ack `msg`
        Message ack_msg(msg.id_, transferrables);
        ack_msg.type_ = detail::ACK;

        if (!transferrables.empty())
        {
            // This must be empty otherwise we'll have an infinite ACK chain
            NEUROPOD_ERROR("[OPE] Transferrables must be empty when sending an `ack` message.");
        }

        // Send the message
        send_message(ack_msg);
    }
};

// Cleanup control channels for the queue with name `control_queue_name`
void cleanup_control_channels(const std::string &control_queue_name);

} // namespace neuropod