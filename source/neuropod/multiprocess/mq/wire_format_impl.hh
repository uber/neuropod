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

#include "neuropod/internal/logging.hh"
#include "neuropod/multiprocess/mq/wire_format.hh"
#include "neuropod/multiprocess/serialization/ipc_serialization.hh"
#include "neuropod/multiprocess/shm/shm_allocator.hh"

namespace neuropod
{

namespace detail
{

// The maximum size of an inline payload
constexpr size_t INLINE_PAYLOAD_SIZE_BYTES = 8192;

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
        char payload[INLINE_PAYLOAD_SIZE_BYTES];

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

} // namespace neuropod
