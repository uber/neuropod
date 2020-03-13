//
// Uber, Inc. (c) 2020
//

#pragma once

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

namespace neuropod
{

// Serialization used for IPC
// This type of serialization has less strict requirements than normal serialization
// because the data is transient and will be written and read in different processes
// on the same machine (so we don't need to worry about things like endianness).
//
// The default implementation just calls into boost serialization. Specializations
// for efficient serialization of other types are included in the other files in this folder
template <typename T>
void ipc_serialize(std::ostream &out, const T &item)
{
    boost::archive::binary_oarchive ar{out};
    ar << item;
}

template <typename T>
void ipc_deserialize(std::istream &in, T &item)
{
    boost::archive::binary_iarchive ar{in};
    ar >> item;
}

// Specialization for uint64_t (for DONE messages)
template <>
void ipc_serialize(std::ostream &out, const uint64_t &data);

template <>
void ipc_deserialize(std::istream &in, uint64_t &data);

} // namespace neuropod