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

#include "neuropod/serialization/serialization.hh"

#include "neuropod/internal/neuropod_tensor.hh"

#include <mutex>
#include <unordered_map>

namespace neuropod
{

namespace
{

std::once_flag                                                     registrar_initialized;
std::unique_ptr<std::unordered_map<std::string, serialize_fn_t>>   registered_serializers;
std::unique_ptr<std::unordered_map<std::string, deserialize_fn_t>> registered_deserializers;

void init_registrar_if_needed()
{
    std::call_once(registrar_initialized, []() {
        registered_serializers   = stdx::make_unique<std::unordered_map<std::string, serialize_fn_t>>();
        registered_deserializers = stdx::make_unique<std::unordered_map<std::string, deserialize_fn_t>>();
    });
}

} // namespace

namespace detail
{

void register_serializer_internal(const std::string &tag, serialize_fn_t serialize_fn, deserialize_fn_t deserialize_fn)
{
    init_registrar_if_needed();
    (*registered_deserializers)[tag] = deserialize_fn;
    (*registered_serializers)[tag]   = serialize_fn;
}

void serialize(boost::archive::binary_oarchive &out, const NeuropodValue &item)
{
    init_registrar_if_needed();
    const auto tag = item.get_serialize_tag();

    // Write the tag
    out << tag;

    auto it = registered_serializers->find(tag);
    if (it == registered_serializers->end())
    {
        NEUROPOD_ERROR("Serialization function not found for tag '{}'", tag);
    }

    // Run the serializer function
    it->second(item, out);
}

template <>
std::shared_ptr<NeuropodValue> deserialize(boost::archive::binary_iarchive &ar, NeuropodTensorAllocator &allocator)
{
    init_registrar_if_needed();
    // Read the tag
    std::string tag;
    ar >> tag;

    auto it = registered_deserializers->find(tag);
    if (it == registered_deserializers->end())
    {
        NEUROPOD_ERROR("Deserialization function not found for tag '{}'", tag);
    }

    // Get the function and run it
    return it->second(ar, allocator);
}

void serialize(boost::archive::binary_oarchive &out, const NeuropodValueMap &item)
{
    int num_items = item.size();
    out << num_items;

    for (const auto &pair : item)
    {
        out << pair.first;
        serialize(out, *pair.second);
    }
}

template <>
NeuropodValueMap deserialize(boost::archive::binary_iarchive &ar, NeuropodTensorAllocator &allocator)
{
    NeuropodValueMap out;
    int              num_items;
    ar >> num_items;

    for (int i = 0; i < num_items; i++)
    {
        std::string key;
        ar >> key;

        out[key] = deserialize<std::shared_ptr<NeuropodValue>>(ar, allocator);
    }

    return out;
}

} // namespace detail

} // namespace neuropod
