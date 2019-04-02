//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/neuropod_tensor.hh"

namespace neuropods
{

// This should be incremented on any breaking changes
static constexpr int SERIALIZATION_VERSION = 1;

static constexpr size_t MAX_NAME_LENGTH = 256;
static constexpr size_t MAX_NUM_DIMS    = 128;

// The structure used to store the serialized tensor
struct serialized_wrapper {
    uint64_t    serialization_version;          // the version of the serialization code
    TensorType  tensor_type;                    // the tensor type
    char        tensor_name[MAX_NAME_LENGTH];   // the tensor name as a null terminated string
    uint64_t    ndims;                          // the number of dimensions
    int64_t     dims[MAX_NUM_DIMS];             // the size of each dimension
    uint64_t    data_size_bytes;                // the number of bytes of data
    uint8_t     data[];                         // the data in the tensor
} __attribute__((packed));

// Utilities
inline size_t get_serialized_byte_length(size_t data_size_bytes)
{
    return sizeof(serialized_wrapper) + data_size_bytes;
}

inline size_t get_serialized_byte_length(const serialized_wrapper &serialized)
{
    return get_serialized_byte_length(serialized.data_size_bytes);
}

// A function that can allocate a serialized_wrapper
using SerializedWrapperAllocator = std::function<std::shared_ptr<serialized_wrapper>(size_t)>;

// A default implementation of a SerializedWrapperAllocator
const auto default_serialized_wrapper_allocator = [](size_t length)
{
    auto wrapper = static_cast<serialized_wrapper *>(malloc(length));
    return std::shared_ptr<serialized_wrapper>(wrapper, [](serialized_wrapper * p) {
        free(p);
    });
};

// A NeuropodTensor that is serializable
template <typename T>
class SerializedNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<std::shared_ptr<serialized_wrapper>>
{
public:
    SerializedNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims, const SerializedWrapperAllocator &allocator = default_serialized_wrapper_allocator)
        : TypedNeuropodTensor<T>(name, dims)
    {
        // Allocate our struct using the user provided SerializedWrapperAllocator
        serialized_ = allocator(get_serialized_byte_length());

        // >= because of null terminated string
        if (name.length() >= MAX_NAME_LENGTH)
        {
            NEUROPOD_ERROR("Tensor names must be less than " << MAX_NAME_LENGTH << " characters long. Tried to create a tensor with a name of length " << name.length() << ": " << name);
        }

        if (dims.size() > MAX_NUM_DIMS)
        {
            NEUROPOD_ERROR("Tensors must not have more than " << MAX_NUM_DIMS << " dimensions. Tried to create a tensor '" << name << "' with " << dims.size() << " dimensions");
        }

        // Populate the header
        serialized_->serialization_version = SERIALIZATION_VERSION;
        serialized_->tensor_type = this->get_tensor_type();

        // Copy up to MAX_NAME_LENGTH characters
        strncpy(serialized_->tensor_name, name.c_str(), MAX_NAME_LENGTH);
        serialized_->ndims = dims.size();
        std::copy(dims.begin(), dims.end(), serialized_->dims);
        serialized_->data_size_bytes = get_data_size_bytes();
    }

    // This tensor type doesn't support wrapping existing memory so we'll do a copy
    SerializedNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims, void * data, const Deleter &deleter)
        : SerializedNeuropodTensor<T>(name, dims)
    {
        // We need to make a copy in order to serialize
        this->copy_from(static_cast<T *>(data), this->get_num_elements());

        // Make sure we run the deleter
        run_deleter(register_deleter(deleter, data));
    }

    // Load a serialized tensor
    // See `deserialize_tensor` below
    SerializedNeuropodTensor(
        const std::string &name,
        const std::vector<int64_t> &dims,
        std::shared_ptr<serialized_wrapper> serialized
    )
        : TypedNeuropodTensor<T>(name, dims),
          serialized_(serialized)
    {
        if (serialized_->serialization_version != SERIALIZATION_VERSION)
        {
            NEUROPOD_ERROR("This serialized tensor was created with a different version of Neuropod serialization code."
                "Expected version " << SERIALIZATION_VERSION << " but got " << serialized_->serialization_version);
        }
    }

    virtual ~SerializedNeuropodTensor() = default;

    // Compute the number of bytes of data in the tensor (not including the header)
    inline size_t get_data_size_bytes()
    {
        return this->get_num_elements() * sizeof(T);
    }

    // Compute the number of bytes required to store this tensor
    inline size_t get_serialized_byte_length()
    {
        return neuropods::get_serialized_byte_length(get_data_size_bytes());
    }

    // Get a pointer to the underlying data
    T *get_raw_data_ptr() {
        void * data = serialized_->data;
        return static_cast<T *>(data);
    }

    const T *get_raw_data_ptr() const {
        void * data = serialized_->data;
        return static_cast<T *>(data);
    }

    std::shared_ptr<serialized_wrapper> get_native_data() {
        return serialized_;
    }

private:
    // A pointer to the serialized_wrapper struct
    std::shared_ptr<serialized_wrapper> serialized_;
};

// A utility function that deserializes a tensor
template <template <class> class TensorClass, typename... Params>
std::shared_ptr<NeuropodTensor> deserialize_tensor(std::shared_ptr<serialized_wrapper> serialized, Params &&... params)
{
    // Make the tensor
    return make_tensor<TensorClass>(
        serialized->tensor_type,
        serialized->tensor_name,
        std::vector<int64_t>(serialized->dims, serialized->dims + serialized->ndims),
        serialized,
        std::forward<Params>(params)...
    );
}

} // namespace neuropods
