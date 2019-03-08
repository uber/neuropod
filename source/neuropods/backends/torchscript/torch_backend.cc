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
        std::stringstream ss;
        ss << "Failed to load graph from path " << graph_path.c_str();
        throw std::runtime_error(ss.str());
    }

    auto model = torch::jit::load(stream);
    if (!model)
    {
        std::stringstream ss;
        ss << "Failed to deserialize graph from path " << graph_path.c_str();
        throw std::runtime_error(ss.str());
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

std::unique_ptr<NeuropodTensor> get_tensor_from_key_value_pair(at::ivalue::Tuple *tuple)
{
    const std::string &name        = tuple->elements()[0].toString()->string();
    auto               tensor      = tuple->elements()[1].toTensor();
    auto               tensor_type = get_neuropod_type_from_torch_type(tensor.scalar_type());
    return make_tensor<TorchNeuropodTensor>(tensor_type, name, tensor);
}

} // namespace

TorchNeuropodBackend::TorchNeuropodBackend(const std::string &neuropod_path, std::unique_ptr<ModelConfig> &model_config)
    : model_(load_model_from_path(get_graph_path(neuropod_path)))
{
}

TorchNeuropodBackend::~TorchNeuropodBackend() = default;

// Run inference
std::unique_ptr<TensorStore> TorchNeuropodBackend::infer(const TensorStore &inputs)
{
    torch::NoGradGuard guard;

    // Get inference schema
    auto &      method = model_->get_method("forward");
    const auto &schema = method.getSchema();

    // Define the vector of inputs and add the inputs
    std::vector<torch::jit::IValue> torch_inputs(schema.arguments().size());
    for (const std::shared_ptr<NeuropodTensor> &tensor : inputs.tensors)
    {
        const auto  input_name = tensor->get_name();
        const auto &input_data
            = std::dynamic_pointer_cast<NativeDataContainer<torch::jit::IValue>>(tensor)->get_native_data();

        auto arg_index = schema.argumentIndexWithName(input_name);
        if (!arg_index.has_value())
        {
            std::stringstream ss;
            ss << "Input '" << input_name.c_str() << "' does not exist. Model inputs " << schema;
            throw std::runtime_error(ss.str());
        }

        // TODO(vip): transfer to the correct device
        // .to(device) is a no-op if the tensor is already transferred
        torch_inputs.at(arg_index.value()) = input_data;
    }

    // Run inference
    c10::IValue result = model_->forward(torch_inputs);

    // Get outputs
    auto to_return = stdx::make_unique<TensorStore>();

    auto outputs_tuple      = result.toTuple();
    bool is_tuple_of_tuples = outputs_tuple->elements()[0].isTuple();
    if (!is_tuple_of_tuples)
    {
        // Just a tuple
        to_return->tensors.emplace_back(get_tensor_from_key_value_pair(outputs_tuple.get()));
    }
    else
    {
        // A tuple of tuples
        for (size_t pos = 0; pos < outputs_tuple->elements().size(); ++pos)
        {
            auto tuple = outputs_tuple->elements()[pos].toTuple();
            to_return->tensors.emplace_back(get_tensor_from_key_value_pair(tuple.get()));
        }
    }

    return to_return;
}

REGISTER_NEUROPOD_BACKEND(TorchNeuropodBackend, "torchscript")

} // namespace neuropods
