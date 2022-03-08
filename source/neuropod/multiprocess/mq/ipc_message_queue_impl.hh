/* Copyright (c) 2020 The Neuropod Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include "neuropod/multiprocess/mq/ipc_message_queue.hh"

namespace ipc = boost::interprocess;

namespace neuropod
{

namespace detail
{

// Used to generate IDs for messages
extern std::atomic_uint64_t msg_counter;

// The max size for the send and recv control queues
constexpr auto MAX_QUEUE_SIZE = 20;

template <typename UserPayloadType>
inline std::unique_ptr<ipc::message_queue> make_queue(const std::string &control_queue_name_, const std::string &suffix)
{
    return stdx::make_unique<ipc::message_queue>(ipc::open_or_create,
                                                 ("neuropod_" + control_queue_name_ + suffix).c_str(),
                                                 MAX_QUEUE_SIZE,
                                                 sizeof(WireFormat<UserPayloadType>));
}

template <typename UserPayloadType>
inline std::unique_ptr<ipc::message_queue> make_send_queue(const std::string &control_queue_name_, ProcessType type)
{
    // Change the suffix depending on if this is the main process or worker process
    return make_queue<UserPayloadType>(control_queue_name_, type == MAIN_PROCESS ? "_tw" : "_fw");
}

template <typename UserPayloadType>
inline std::unique_ptr<ipc::message_queue> make_recv_queue(const std::string &control_queue_name_, ProcessType type)
{
    // Change the suffix depending on if this is the main process or worker process
    return make_queue<UserPayloadType>(control_queue_name_, type == WORKER_PROCESS ? "_tw" : "_fw");
}

} // namespace detail

// The worker loop for the message reading thread
template <typename UserPayloadType>
void IPCMessageQueue<UserPayloadType>::read_worker_loop()
{
    while (true)
    {
        // Compute the timeout
        auto timeout_at = boost::interprocess::microsec_clock::universal_time() +
                          boost::posix_time::milliseconds(detail::MESSAGE_TIMEOUT_MS);

        // Get a message
        auto         received = stdx::make_unique<WireFormat>();
        size_t       received_size;
        unsigned int priority;
        bool         successful_read =
            recv_queue_->timed_receive(received.get(), sizeof(WireFormat), received_size, priority, timeout_at);

        if (!successful_read)
        {
            // We timed out
            SPDLOG_ERROR("Timed out waiting for a response from worker process. "
                         "Didn't receive a message in {}ms, but expected a heartbeat every {}ms.",
                         detail::MESSAGE_TIMEOUT_MS,
                         detail::HEARTBEAT_INTERVAL_MS);

            // Set a flag so that any pending writers know
            lost_heartbeat_.store(true, std::memory_order_relaxed);

            // Let any pending readers know
            // (Because recv isn't threadsafe, there should only be one)
            out_queue_.try_emplace(nullptr);

            // Shutdown this thread
            break;
        }

        if (received->type == detail::USER_PAYLOAD)
        {
            SPDLOG_TRACE("OPE: Read thread received user payload {}.", received->payload_type);
        }
        else
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

            transferrable_controller_->done(acked_id);
        }
        else if (received->type == detail::SHUTDOWN_QUEUES)
        {
            // Start a shutdown.
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
            // Only shutdown once we've received DONEs for all the messages we've sent
            const auto in_transit_count = transferrable_controller_->size();
            if (in_transit_count == 0)
            {
                // We can finish shutting down
                break;
            }
            else
            {
                SPDLOG_TRACE("OPE: Tried to shut down read worker thread, but still waiting on {} `DONE` messages.",
                             in_transit_count);
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

    // Send w/ timeout + heartbeat check
    while (true)
    {
        auto timeout_at = boost::interprocess::microsec_clock::universal_time() +
                          boost::posix_time::milliseconds(detail::HEARTBEAT_INTERVAL_MS);

        if (send_queue_->timed_send(&msg, sizeof(msg), 0, timeout_at))
        {
            // Successfully sent
            break;
        }

        // Make sure the worker process is still alive
        throw_if_lost_heartbeat();
    }
}

// Throw an error if we lost communication with the other process
template <typename UserPayloadType>
void IPCMessageQueue<UserPayloadType>::throw_if_lost_heartbeat()
{
    if (lost_heartbeat_.load(std::memory_order_relaxed))
    {
        // We lost the heartbeat so let's throw an error here (where the user can catch it)
        NEUROPOD_ERROR("OPE lost communication with the other process. See logs for more details.");
    }
}

template <typename UserPayloadType>
IPCMessageQueue<UserPayloadType>::IPCMessageQueue(const std::string &control_queue_name, ProcessType type)
    : out_queue_(detail::MAX_QUEUE_SIZE),
      control_queue_name_(control_queue_name),
      send_queue_(detail::make_send_queue<UserPayloadType>(control_queue_name_, type)),
      recv_queue_(detail::make_recv_queue<UserPayloadType>(control_queue_name_, type)),
      heartbeat_controller_(stdx::make_unique<detail::HeartbeatController>(*this)),
      transferrable_controller_(stdx::make_unique<detail::TransferrableController>()),
      lost_heartbeat_(false),
      read_worker_(&IPCMessageQueue<UserPayloadType>::read_worker_loop, this)
{
}

template <typename UserPayloadType>
IPCMessageQueue<UserPayloadType>::~IPCMessageQueue()
{
    heartbeat_controller_.reset();

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
        transferrable_controller_->add(msg.id, transferrables);
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
        transferrable_controller_->add(msg.id, transferrables);
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
QueueMessage<UserPayloadType> IPCMessageQueue<UserPayloadType>::recv_message()
{
    // Make sure the worker process is still alive
    throw_if_lost_heartbeat();

    // Read a message
    std::unique_ptr<WireFormat> out;
    out_queue_.pop(out);

    if (out == nullptr)
    {
        // We lost communication with the other processs while we were reading
        NEUROPOD_ERROR("OPE lost communication with the other process. See logs for more details.");
    }

    SPDLOG_TRACE(
        "OPE: Received user payload of type: {} (requires done: {})", out->payload_type, out->requires_done_msg);

    // Convert this to a shared ptr with a deleter that acks the message
    auto                        shared_this = this->shared_from_this();
    std::shared_ptr<WireFormat> received_shared(out.release(), [shared_this](WireFormat *msg) {
        if (msg->requires_done_msg)
        {
            // Notify the other process that this message is done being read from
            // and any associated resources can be freed

            // Create a message to ack `msg`
            WireFormat ack_msg;
            ack_msg.type = detail::DONE;

            // Serialize the payload
            detail::Transferrables transferrables;
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

    return QueueMessage<UserPayloadType>(std::move(received_shared));
}

} // namespace neuropod
