//
// Uber, Inc. (c) 2020
//

#pragma once

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/multiprocess/control_messages.hh"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

#include <unordered_map>

namespace neuropod
{

// This type of serialization has less strict requirements because
// the data is transient and will be written and read in different processes
// on the same machine.
// The default implementation just calls into boost serialization
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

// Specializations for NeuropodValueMap
template <>
void ipc_serialize(std::ostream &out, const NeuropodValueMap &data);

template <>
void ipc_deserialize(std::istream &in, NeuropodValueMap &data);

// Specializations for ope_load_config
// Because of a bug in boost serialization, we can't just mark that struct
// as serializable
template <>
void ipc_serialize(std::ostream &out, const ope_load_config &data);

template <>
void ipc_deserialize(std::istream &in, ope_load_config &data);

// Specialization for uint64_t (for ACK messages)
// Serialization and deserialization are running on the same machine so we don't
// need to worry about things like endianness
template <>
void ipc_serialize(std::ostream &out, const uint64_t &data);

template <>
void ipc_deserialize(std::istream &in, uint64_t &data);

} // namespace neuropod