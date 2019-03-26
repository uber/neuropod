//
// Uber, Inc. (c) 2018
//

#include "torch_backend.hh"

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "neuropods/backends/torchscript/type_utils.hh"
#include "neuropods/internal/tensor_store.hh"

namespace neuropods
{

namespace
{

std::shared_ptr<torch::jit::script::Module> load_model_from_path(const std::string &graph_path)
{
    std::ifstream stream(graph_path, std::ios_base::binary);
    if (!stream.good())
    {
        NEUROPOD_ERROR("Failed to load graph from path " << graph_path.c_str());
    }

    auto model = torch::jit::load(stream);
    if (!model)
    {
        NEUROPOD_ERROR("Failed to deserialize graph from path " << graph_path.c_str());
    }
    return model;
}

// Get graph path from a neuropod path
std::string get_graph_path(const std::string &neuropod_path)
{
    if (neuropod_path.back() == '/')
    {
        return neuropod_path + "0/data/model.pt";
    }

    return neuropod_path + "/0/data/model.pt";
}

} // namespace

TorchNeuropodBackend::TorchNeuropodBackend(const std::string &neuropod_path, std::unique_ptr<ModelConfig> &model_config)
    : model_(load_model_from_path(get_graph_path(neuropod_path)))
{
}

TorchNeuropodBackend::~TorchNeuropodBackend() = default;

// Run inference
std::unique_ptr<TensorStore> TorchNeuropodBackend::infer(const std::unordered_set<std::shared_ptr<NeuropodTensor>> &inputs)
{
    torch::NoGradGuard guard;

    // Get inference schema
    auto &      method = model_->get_method("forward");
    const auto &schema = method.getSchema();

    // Define the vector of inputs and add the inputs
    std::vector<torch::jit::IValue> torch_inputs(schema.arguments().size());
    for (const std::shared_ptr<NeuropodTensor> &tensor : inputs)
    {
        const auto  input_name = tensor->get_name();
        const auto &input_data
            = std::dynamic_pointer_cast<NativeDataContainer<torch::jit::IValue>>(tensor)->get_native_data();

        auto arg_index = schema.argumentIndexWithName(input_name);
        if (!arg_index.has_value())
        {
            NEUROPOD_ERROR("Input '" << input_name.c_str() << "' does not exist. Model inputs " << schema);
        }

        // TODO(vip): transfer to the correct device
        // .to(device) is a no-op if the tensor is already transferred
        torch_inputs.at(arg_index.value()) = input_data;
    }

    // Run inference
    c10::IValue result = model_->forward(torch_inputs);

    // Get outputs
    auto to_return = stdx::make_unique<TensorStore>();

    const auto &outputs_dict = result.toGenericDict()->elements();
    for (const auto &elem : outputs_dict)
    {
        // Get the name of the tensor
        const std::string &name = elem.first.toString()->string();

        if (elem.second.isGenericList())
        {
            // A list of strings
            // This is used in place of string tensors because torch does not
            // have native support for string tensors
            auto tensor = elem.second;

            // Make sure it's actually a list of strings (or empty)
            auto &list = tensor.toGenericListRef();
            if (list.size() != 0 && !list[0].isString())
            {
                NEUROPOD_ERROR("Neuropod got a list of type '" << list[0].tagKind() << "' for tensor '" << name << "'."
                    "Only tensors or lists of strings are supported");
            }

            // Make a TorchNeuropodTensor
            auto neuropod_tensor = stdx::make_unique<TorchNeuropodTensor<std::string>>(name, tensor);

            // Add it to our TensorStore
            to_return->tensors.emplace_back(std::move(neuropod_tensor));
        }
        else if (elem.second.isTensor())
        {
            // Torch tensor
            auto tensor = elem.second.toTensor();

            // Get the type and make a TorchNeuropodTensor
            auto tensor_type = get_neuropod_type_from_torch_type(tensor.scalar_type());
            auto neuropod_tensor = make_tensor<TorchNeuropodTensor>(tensor_type, name, tensor);

            // Add it to our TensorStore
            to_return->tensors.emplace_back(std::move(neuropod_tensor));
        }
        else
        {
            NEUROPOD_ERROR("Neuropod returned an invalid type! All outputs must be tensors"
                "or lists of strings. Got type '" << elem.second.tagKind() << "' for tensor '" << name << "'");
        }
    }

    return to_return;
}

REGISTER_NEUROPOD_BACKEND(TorchNeuropodBackend, "torchscript")

} // namespace neuropods
