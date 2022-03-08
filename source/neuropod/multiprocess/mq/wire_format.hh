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

#include "neuropod/multiprocess/mq/transferrables.hh"

namespace neuropod
{

namespace detail
{

// The on-the-wire format of the data
// UserPayloadType should be an enum that specifies types of user payloads
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

} // namespace neuropod

#include "neuropod/multiprocess/mq/wire_format_impl.hh"
