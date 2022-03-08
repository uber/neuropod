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

#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/memory_utils.hh"

#include <boost/pfr/precise.hpp>

#include <istream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuropod
{

namespace detail
{

// Utilities for checked reads and writes to and from streams
template <typename... Params>
inline void checked_write(std::ostream &stream, Params &&... params)
{
    stream.write(std::forward<Params>(params)...);
    if (!stream)
    {
        NEUROPOD_ERROR("Writing to stream failed during IPC serialization");
    }
}

template <typename... Params>
inline void checked_read(std::istream &stream, Params &&... params)
{
    stream.read(std::forward<Params>(params)...);
    if (!stream)
    {
        NEUROPOD_ERROR("Reading from stream failed during IPC serialization");
    }
}

} // namespace detail

// Serialization used for IPC
// This type of serialization has less strict requirements than normal serialization
// because the data is transient and will be written and read in different processes
// on the same machine (so we don't need to worry about things like endianness).
//
// These methods handle primitive types (other than bool)
template <typename T>
inline void ipc_serialize(std::ostream &out, const T &item)
{
    constexpr bool is_integral  = std::is_integral<T>::value;
    constexpr bool is_aggregate = std::is_aggregate<T>::value;

    static_assert(is_integral || is_aggregate, "The ipc_serialize function must be specialized for the requested type");

    if constexpr (is_integral)
    {
        // Primitive types
        detail::checked_write(out, reinterpret_cast<const char *>(&item), sizeof(item));
    }
    else if (is_aggregate)
    {
        // Structs
        boost::pfr::for_each_field(item, [&](auto &field) { ipc_serialize(out, field); });
    }
}

template <typename T>
inline void ipc_deserialize(std::istream &in, T &item)
{
    constexpr bool is_integral  = std::is_integral<T>::value;
    constexpr bool is_aggregate = std::is_aggregate<T>::value;

    static_assert(is_integral || is_aggregate,
                  "The ipc_deserialize function must be specialized for the requested type");

    if constexpr (is_integral)
    {
        // Primitive types
        detail::checked_read(in, reinterpret_cast<char *>(&item), sizeof(item));
    }
    else if (is_aggregate)
    {
        // Structs
        boost::pfr::for_each_field(item, [&](auto &field) { ipc_deserialize(in, field); });
    }
}

// Specialization for bool
template <>
inline void ipc_serialize(std::ostream &out, const bool &item)
{
    uint8_t value = item;
    ipc_serialize(out, value);
}

template <>
inline void ipc_deserialize(std::istream &in, bool &item)
{
    uint8_t value;
    ipc_deserialize(in, value);
    item = value;
}

// Specialization for strings
template <>
inline void ipc_serialize(std::ostream &out, const std::string &item)
{
    // Write the length
    const size_t length = item.length();
    ipc_serialize(out, length);

    // Write the content
    detail::checked_write(out, item.c_str(), length);
}

template <>
inline void ipc_deserialize(std::istream &in, std::string &item)
{
    // A thread local buffer used for reading in strings of less than a certain size
    static thread_local char static_buffer[2048];

    // Get the length
    size_t length;
    ipc_deserialize(in, length);

    // Read the string into a buffer
    if (length < sizeof(static_buffer))
    {
        detail::checked_read(in, static_buffer, length);

        // Set the content of `item`
        item.assign(static_buffer, length);
    }
    else
    {
        // We need to allocate a larger buffer
        auto buffer = stdx::make_unique<char[]>(length);
        detail::checked_read(in, buffer.get(), length);

        // Set the content of `item`
        item.assign(buffer.get(), length);
    }
}

// Specialization for serializing char[] (for constant strings)
template <size_t N>
inline void ipc_serialize(std::ostream &out, const char (&item)[N])
{
    std::string str = item;
    ipc_serialize(out, str);
}

// Specialization for vector
template <typename T>
inline void ipc_serialize(std::ostream &out, const std::vector<T> &item)
{
    // Write the size
    const size_t size = item.size();
    ipc_serialize(out, size);

    // Write the content
    for (const auto &elem : item)
    {
        ipc_serialize(out, elem);
    }
}

template <typename T>
inline void ipc_deserialize(std::istream &in, std::vector<T> &item)
{
    // Get the size
    size_t size;
    ipc_deserialize(in, size);

    // Get the content
    for (int i = 0; i < size; i++)
    {
        T elem;
        ipc_deserialize(in, elem);
        item.emplace_back(std::move(elem));
    }
}

// Specialization for unordered_map
template <typename K, typename V>
inline void ipc_serialize(std::ostream &out, const std::unordered_map<K, V> &item)
{
    // Write the number of items
    size_t num_items = item.size();
    ipc_serialize(out, num_items);

    for (const auto &entry : item)
    {
        // Write the key
        ipc_serialize(out, entry.first);

        // Write the value
        ipc_serialize(out, entry.second);
    }
}

template <typename K, typename V>
inline void ipc_deserialize(std::istream &in, std::unordered_map<K, V> &item)
{
    // Read the number of items
    size_t num_items;
    ipc_deserialize(in, num_items);

    for (int i = 0; i < num_items; i++)
    {
        // Read the key
        K key;
        ipc_deserialize(in, key);

        // Read the value
        V value;
        ipc_deserialize(in, value);
        item[key] = std::move(value);
    }
}

} // namespace neuropod
