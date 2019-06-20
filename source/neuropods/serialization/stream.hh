//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/backends/tensor_allocator.hh"

#include <cstring>

namespace neuropods
{
namespace
{

constexpr size_t MAX_NAME_LENGTH = 240;
constexpr size_t MAX_RANK = 16;
struct NeuropodTensorHeader
{
    char magic_bytes[8];
    uint32_t version;
    uint32_t type;
    char name[MAX_NAME_LENGTH + 1];
    uint8_t rank;
    int64_t dims[MAX_RANK];
} __attribute__((packed));

size_t strnlen(const char *str, size_t max)
{
    const char *end = static_cast<const char*>(memchr(str, 0, max));
    return end ? (size_t)(end - str) : max;
}

struct serialize_visitor : public NeuropodTensorVisitor<void>
{
    template <typename T, class OStream>
    void operator()(const TypedNeuropodTensor<T> *tensor, OStream & ostream) const
    {
        ostream.write(reinterpret_cast<const char*>(tensor->get_raw_data_ptr()), sizeof(T) * tensor->get_num_elements());
    }

    template <class OStream>
    void operator()(const TypedNeuropodTensor<std::string> *tensor, OStream & ostream) const
    {
        auto data = tensor->get_data_as_vector();
        for (const std::string& element : data)
        {
            int32_t size = element.size();
            ostream.write(reinterpret_cast<char*>(&size), sizeof(size));
            ostream.write(element.c_str(), size);
        }
    }
};

struct deserialize_visitor : public NeuropodTensorVisitor<void>
{
    template <typename T, class IStream>
    void operator()(TypedNeuropodTensor<T> *tensor, IStream & istream) const
    {
        istream.read(reinterpret_cast<char*>(tensor->get_raw_data_ptr()), sizeof(T) * tensor->get_num_elements());
    }

    template <class IStream>
    void operator()(TypedNeuropodTensor<std::string> *tensor, IStream & istream) const
    {
        std::vector<std::string> data;
        data.reserve(tensor->get_num_elements());
        std::vector<char> buffer;
        for (int i = 0; i < tensor->get_num_elements(); ++i)
        {
            int32_t size;
            istream.read(reinterpret_cast<char*>(&size), sizeof(size));
            buffer.resize(size);
            
            istream.read(&buffer[0], size);
            std::string buffer_as_string(&buffer[0], size);
            data.emplace_back(std::move(buffer_as_string));
        }
        tensor->set(std::move(data));
    }
};

} // anonymous namespace

// This should be incremented on any breaking changes
static const char* MAGIC_HEADER = "NEUPTNSR";
static const uint32_t SERIALIZATION_VERSION = 1;

template<class OStream>
void serialize_tensor(OStream & ostream, const std::string &name, const NeuropodTensor &tensor)
{
    NeuropodTensorHeader header;
    std::strncpy(header.magic_bytes, MAGIC_HEADER, sizeof(NeuropodTensorHeader::name));
    header.version = SERIALIZATION_VERSION;
    header.type = static_cast<int32_t>(tensor.get_tensor_type());
    if (name.size() > MAX_NAME_LENGTH)
    {
        NEUROPOD_ERROR("Maximum allowed tensor name length is " << MAX_NAME_LENGTH << ". Tensor that violates this constraint: '" << name << "'.");
    }
    std::strcpy(header.name, name.c_str());

    const auto& dims = tensor.get_dims();
    if (dims.size() > MAX_RANK)
    {
        NEUROPOD_ERROR("Maximum rank for a serializable tensor is " << MAX_RANK << ". Tensor '" << name << "' rank is: " << dims.size());
    }

    header.rank = dims.size();
    std::copy(dims.begin(), dims.end(), &header.dims[0]);

    ostream.write(reinterpret_cast<char*>(&header), sizeof(header));

    tensor.apply_visitor(serialize_visitor{}, ostream);
}

template<class IStream>
void deserialize_tensor(IStream & istream, neuropods::NeuropodTensorAllocator &allocator, std::string &name, std::shared_ptr<NeuropodTensor> &out)
{
    NeuropodTensorHeader header;
    
    istream.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (std::strncmp(MAGIC_HEADER, header.magic_bytes, sizeof(*MAGIC_HEADER)) != 0)
    {
        NEUROPOD_ERROR("Did not find neuropod magic header. Deserializing from a corrupt data stream.");
    }

    if (header.version != SERIALIZATION_VERSION)
    {
        NEUROPOD_ERROR("This serialized tensor was created with a different version of Neuropod serialization code."
            "Expected version " << SERIALIZATION_VERSION << " but got " << header.version);
    }

    const size_t name_size = strnlen(header.name, sizeof NeuropodTensorHeader::name);

    name.assign(header.name, header.name + name_size);

    std::vector<int64_t> dims;
    if (header.rank > MAX_RANK)
    {
        NEUROPOD_ERROR("Rank is expected to be less then " << MAX_RANK << ". For tensor '" << header.name << " the rank is " << header.rank);
    }
    
    std::copy(&header.dims[0], &header.dims[header.rank], std::insert_iterator<std::vector<int64_t>>(dims, dims.begin()));

    out = allocator.allocate_tensor(dims, static_cast<TensorType>(header.type));
    out->apply_visitor(deserialize_visitor{}, istream);
}

} // namespace neuropods
