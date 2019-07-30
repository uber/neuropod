//
// Uber, Inc. (c) 2019
//

#include "neuropods/backends/tensor_allocator.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/serialization/serialization.hh"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>

namespace neuropods
{

namespace
{

struct serialize_visitor : public NeuropodTensorVisitor<void>
{
    template <typename T, class Archive>
    void operator()(const TypedNeuropodTensor<T> *tensor, Archive &ar) const
    {
        ar &boost::serialization::make_array(tensor->get_raw_data_ptr(), tensor->get_num_elements());
    }

    template <class Archive>
    void operator()(const TypedNeuropodTensor<std::string> *tensor, Archive &ar) const
    {
        const auto data = tensor->get_data_as_vector();
        ar &       data;
    }
};

struct deserialize_visitor : public NeuropodTensorVisitor<void>
{
    template <typename T, class Archive>
    void operator()(TypedNeuropodTensor<T> *tensor, Archive &ar) const
    {
        ar &boost::serialization::make_array(tensor->get_raw_data_ptr(), tensor->get_num_elements());
    }

    template <class Archive>
    void operator()(TypedNeuropodTensor<std::string> *tensor, Archive &ar) const
    {
        std::vector<std::string> data;
        ar &                     data;
        tensor->set(data);
    }
};

void serialize_tensor(const NeuropodTensor &tensor, boost::archive::binary_oarchive &ar)
{
    int tensor_type = static_cast<int>(tensor.get_tensor_type());
    ar << tensor_type;
    ar << tensor.get_dims();
    tensor.apply_visitor(serialize_visitor{}, ar);
}

std::shared_ptr<NeuropodTensor> deserialize_tensor(boost::archive::binary_iarchive &ar,
                                                   NeuropodTensorAllocator &        allocator)
{
    int                  type;
    std::vector<int64_t> dims;
    ar >> type;
    ar >> dims;

    auto out = allocator.allocate_tensor(dims, static_cast<TensorType>(type));
    out->apply_visitor(deserialize_visitor{}, ar);
    return out;
}

} // namespace

MAKE_SERIALIZABLE(NeuropodTensor, serialize_tensor, deserialize_tensor);

} // namespace neuropods
