//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/internal/error_utils.hh"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <functional>
#include <memory>
#include <string>

namespace neuropods
{

// Forward declarations
class NeuropodValue;
class NeuropodTensorAllocator;

// Types for serialization and deserialization functions
using serialize_fn_t = std::function<void(const NeuropodValue &, boost::archive::binary_oarchive &)>;
using deserialize_fn_t =
    std::function<std::shared_ptr<NeuropodValue>(boost::archive::binary_iarchive &, NeuropodTensorAllocator &)>;

namespace detail
{

void register_serializer_internal(const std::string &tag, serialize_fn_t serialize_fn, deserialize_fn_t deserialize_fn);

// Register a class as serializable. The class should inherit from NeuropodValue
// and should use the SET_SERIALIZE_TAG macro in its definition
// This function is used in the MAKE_SERIALIZABLE macro below
template <typename T>
bool register_serializable(std::function<void(const T &, boost::archive::binary_oarchive &)> serialize_fn,
                           deserialize_fn_t                                                  deserialize_fn)
{
    register_serializer_internal(
        T::get_static_serialize_tag(),
        [serialize_fn](const NeuropodValue &val, boost::archive::binary_oarchive &out) {
            serialize_fn(dynamic_cast<const T &>(val), out);
        },
        deserialize_fn);

    return true;
}

// Serialize a NeuropodValue
void serialize(boost::archive::binary_oarchive &out, const NeuropodValue &item);

// Deserialize from an archive
std::shared_ptr<NeuropodValue> deserialize(boost::archive::binary_iarchive &ar, NeuropodTensorAllocator &allocator);

} // namespace detail

namespace
{

// This should be incremented on any breaking changes
static constexpr int SERIALIZATION_VERSION = 1;

} // namespace

// A function to serialize to an archive along with a serialization version
template <typename... Params>
void serialize(std::ostream &out, Params &&... params)
{
    boost::archive::binary_oarchive ar{out};
    ar << SERIALIZATION_VERSION;
    detail::serialize(ar, std::forward<Params>(params)...);
}

// A function to read from an archive that starts with the serialization version
template <typename... Params>
std::shared_ptr<NeuropodValue> deserialize(std::istream &in, Params &&... params)
{
    boost::archive::binary_iarchive ar{in};
    int                             version;
    ar >> version;
    if (version != SERIALIZATION_VERSION)
    {
        NEUROPOD_ERROR("This serialized tensor was created with a different version of Neuropod serialization code."
                       "Expected version "
                       << SERIALIZATION_VERSION << " but got " << version);
    }

    return detail::deserialize(ar, std::forward<Params>(params)...);
}

// Utility to register serializable types
// Note that the type passed in must use the SET_SERIALIZE_TAG macro in its definition
#define MAKE_SERIALIZABLE(CLS, serialize_fn, deserialize_fn) \
    bool is_registered_##CLS = detail::register_serializable<CLS>(serialize_fn, deserialize_fn);

} // namespace neuropods
