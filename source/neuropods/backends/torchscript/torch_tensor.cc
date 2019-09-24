//
// Uber, Inc. (c) 2019
//

#include "torch_tensor.hh"

namespace neuropods
{

TorchNeuropodValue::TorchNeuropodValue(torch::jit::IValue &item)
    : NeuropodValue(false), item_(item)
{}

TorchNeuropodValue::~TorchNeuropodValue() = default;

torch::jit::IValue TorchNeuropodValue::get_native_data() { return item_; }


// Utility function to get an IValue from a torch tensor
torch::jit::IValue get_ivalue_from_torch_tensor(const std::shared_ptr<NeuropodValue> &tensor)
{
    return std::dynamic_pointer_cast<NativeDataContainer<torch::jit::IValue>>(tensor)->get_native_data();
}

} // namespace neuropods
