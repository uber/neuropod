//
// Uber, Inc. (c) 2018
//

#include "tensor_store.hh"

#include <stdexcept>

namespace neuropods
{

std::shared_ptr<NeuropodTensor> TensorStore::find(const std::string &name)
{
    // Since the number of input/output tensors used with a model
    // is usually relatively small, looping through all the tensors
    // should be fast enough
    for (const auto &tensor : tensors)
    {
        if (tensor->get_name() == name)
        {
            return tensor;
        }
    }

    return nullptr;
}

std::shared_ptr<NeuropodTensor> TensorStore::find_or_throw(const std::string &name)
{
    auto tensor = find(name);
    if (!tensor)
    {
        NEUROPOD_ERROR("Tensor not found for name '" << name << "'!");
    }
    return tensor;
}

} // namespace neuropods
