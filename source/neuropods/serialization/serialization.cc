//
// Uber, Inc. (c) 2019
//

#include "neuropods/serialization/serialization.hh"

#include "neuropods/internal/neuropod_tensor.hh"

#include <mutex>
#include <unordered_map>

namespace neuropods
{

namespace
{

std::once_flag registrar_initialized;
std::unordered_map<std::string, serialize_fn_t>*     registered_serializers   = nullptr;
std::unordered_map<std::string, deserialize_fn_t>*   registered_deserializers = nullptr;

void init_registrar_if_needed()
{
    std::call_once(registrar_initialized, [](){
        registered_serializers   = new std::unordered_map<std::string, serialize_fn_t>();
        registered_deserializers = new std::unordered_map<std::string, deserialize_fn_t>();
    });
}

} // namespace

namespace detail
{

void register_serializer_internal(const std::string &tag, serialize_fn_t serialize_fn, deserialize_fn_t deserialize_fn)
{
    init_registrar_if_needed();
    (*registered_deserializers)[tag] = deserialize_fn;
    (*registered_serializers)[tag] = serialize_fn;
}

void serialize(boost::archive::binary_oarchive &out, const NeuropodValue &item)
{
    const auto tag = item.get_serialize_tag();

    // Write the tag
    out << tag;

    auto it = registered_serializers->find(tag);
    if (it == registered_serializers->end())
    {
        NEUROPOD_ERROR("Serialization function not found for tag '" << tag << "'! ");
    }

    // Run the serializer function
    registered_serializers->at(tag)(item, out);
}

std::shared_ptr<NeuropodValue> deserialize(boost::archive::binary_iarchive & ar, NeuropodTensorAllocator &allocator)
{
    // Read the tag
    std::string tag;
    ar >> tag;

    auto it = registered_deserializers->find(tag);
    if (it == registered_deserializers->end())
    {
        NEUROPOD_ERROR("Deserialization function not found for tag '" << tag << "'! ");
    }

    // Get the function and run it
    return registered_deserializers->at(tag)(ar, allocator);
}

} // namespace detail

} // namespace neuropods
