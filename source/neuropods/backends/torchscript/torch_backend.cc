//
// Uber, Inc. (c) 2018
//

#include "torch_backend.hh"

#include "neuropods/backends/torchscript/type_utils.hh"
#include "neuropods/internal/tensor_types.hh"

#include <caffe2/core/macros.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dlfcn.h>

namespace neuropods
{

namespace
{

std::shared_ptr<torch::jit::script::Module> load_model_from_path(const std::string &             graph_path,
                                                                 const std::vector<std::string> &custom_op_paths,
                                                                 const torch::Device &           device)
{
    // Load custom ops
    // TODO(vip): Add a flag allowing users to opt out of loading custom ops

#ifndef __APPLE__
// We need to do this so the custom ops can see the symbols from torch
// This binary is already linked against `libtorch.so`; the dlopen just
// promotes it to RTLD_GLOBAL.
#if CAFFE2_NIGHTLY_VERSION >= 20190601
    void *libtorch = dlopen("libtorch.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
#else
    void *libtorch = dlopen("libtorch.so.1", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
#endif

    if (libtorch == nullptr)
    {
        NEUROPOD_ERROR("Failed to promote libtorch to RTLD_GLOBAL. Error from dlopen: " << dlerror());
    }
#endif

    for (const auto &path : custom_op_paths)
    {
        if (dlopen(path.c_str(), RTLD_NOW) == nullptr)
        {
            NEUROPOD_ERROR("Failed to load custom op. Error from dlopen: " << dlerror());
        }
    }

    std::ifstream stream(graph_path, std::ios_base::binary);
    if (!stream.good())
    {
        NEUROPOD_ERROR("Failed to load graph from path " << graph_path.c_str());
    }

#if CAFFE2_NIGHTLY_VERSION >= 20190717
    auto model = std::make_shared<torch::jit::script::Module>(torch::jit::load(stream, device));
#else
    auto model = torch::jit::load(stream, device);
#endif

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

// Get a custom op path
std::string get_custom_op_path(const std::string &neuropod_path, const std::string &op_basename)
{
    if (neuropod_path.back() == '/')
    {
        return neuropod_path + "0/ops/" + op_basename;
    }

    return neuropod_path + "/0/ops/" + op_basename;
}

std::vector<std::string> get_custom_ops_from_model_config(const std::string &neuropod_path,
                                                          const ModelConfig &model_config)
{
    std::vector<std::string> out;
    for (const auto &item : model_config.custom_ops)
    {
        out.emplace_back(get_custom_op_path(neuropod_path, item));
    }

    return out;
}

// insert IValue to the output map at key with some type validation
void insert_value_in_output(NeuropodValueMap & output,
                            const std::string  name,
                            const c10::IValue &value,
                            const bool         has_type    = false,
                            const TensorType   tensor_type = FLOAT_TENSOR)
{
    if (value.isTensor())
    {
        // Torch tensor
        // Transfer it to CPU
        // .to(device) is a no-op if the tensor is already transferred
        auto tensor = value.toTensor().to(torch::kCPU);

        // Get the type and make a TorchNeuropodTensor
        auto tensor_type     = get_neuropod_type_from_torch_type(tensor.scalar_type());
        auto neuropod_tensor = make_tensor<TorchNeuropodTensor>(tensor_type, tensor);

        // Add it to our output
        output[name] = std::move(neuropod_tensor);
    }
    else if (value.isGenericList())
    {
        // A list of strings
        // This is used in place of string tensors because torch does not
        // have native support for string tensors
        auto &tensor = value;

        const auto &list = tensor.toGenericListRef();

        // if tensor_type string or no tensor_type and empty list or list containing actual string
        if ((has_type && tensor_type == TensorType::STRING_TENSOR) || (!has_type && list.size() == 0) ||
            (!has_type && list[0].isString()))
        {
            // Make a TorchNeuropodTensor
            auto neuropod_tensor = stdx::make_unique<TorchNeuropodTensor<std::string>>(tensor);

            // Add it to our output
            output[name] = std::move(neuropod_tensor);
        }
        // it was bad spec or contained non-string type
        else
        {
            NEUROPOD_ERROR("Neuropod got a list of type '" << list[0].tagKind() << "' for tensor '" << name
                                                           << "'."
                                                              "Only tensors or lists of strings are supported");
        }
    }
    else
    {
        NEUROPOD_ERROR("Neuropod returned an invalid type! All outputs must be tensors"
                       "or lists of strings. Got type '"
                       << value.tagKind() << "' for tensor '" << name << "'");
    }
}

torch::jit::IValue maybe_set_device(const torch::jit::IValue &item, const torch::Device &device)
{
    if (item.isTensor())
    {
        // .to(device) is a no-op if the tensor is already transferred
        return item.toTensor().to(device);
    }

    return item;
}

} // namespace

TorchNeuropodBackend::TorchNeuropodBackend(const std::string &           neuropod_path,
                                           std::unique_ptr<ModelConfig> &model_config,
                                           const RuntimeOptions &        options)
    : options_(options), input_device_mapping_(model_config->input_tensor_device)
{
    // Load the model
    model_ = load_model_from_path(get_graph_path(neuropod_path),
                                  get_custom_ops_from_model_config(neuropod_path, *model_config),

                                  // Load the model onto the appropriate device (ideally a GPU if we have one available)
                                  // Note: this uses the options set in the initializer list above
                                  get_torch_device(DeviceType::GPU));

    for (const auto &tensor_spec : model_config->outputs)
    {
        output_specs_.emplace_back(tensor_spec);
    }
}

TorchNeuropodBackend::TorchNeuropodBackend(const std::string &torchscript_model_path)
    : TorchNeuropodBackend(torchscript_model_path, {})
{
}

TorchNeuropodBackend::TorchNeuropodBackend(const std::string &             torchscript_model_path,
                                           const std::vector<std::string> &custom_op_paths)
    : model_(load_model_from_path(torchscript_model_path, custom_op_paths, torch::kCUDA))
{
}

TorchNeuropodBackend::~TorchNeuropodBackend() = default;

torch::Device TorchNeuropodBackend::get_torch_device(neuropods::DeviceType target_device)
{
    if (options_.visible_device == Device::CPU || !torch::cuda::is_available())
    {
        // No matter what the target device is, we don't have a choice other than running on CPU
        // TODO(vip): warn if visible_device is set to a GPU but CUDA isn't available
        return torch::kCPU;
    }

    if (target_device == DeviceType::CPU)
    {
        return torch::kCPU;
    }
    else
    {
        return torch::Device(torch::kCUDA, options_.visible_device);
    }
}

#if CAFFE2_NIGHTLY_VERSION >= 20190717
#define MAKE_DICT(name) c10::impl::GenericDict name((c10::impl::deprecatedUntypedDict()));
#elif CAFFE2_NIGHTLY_VERSION >= 20190601
#define MAKE_DICT(name) auto name = c10::make_dict<torch::jit::IValue, torch::jit::IValue>();
#else
#define MAKE_DICT(name) torch::ivalue::UnorderedMap name;
#endif

#if CAFFE2_NIGHTLY_VERSION >= 20190717
#define SCHEMA(method) method.function().getSchema()
#else
#define SCHEMA(method) method.getSchema()
#endif

#if CAFFE2_NIGHTLY_VERSION >= 20190601
#define KEY(elem) (elem.key())
#define VALUE(elem) (elem.value())
#define DICT_INSERT(dict, key, value) dict.insert(key, value);
#else
#define KEY(elem) (elem.first)
#define VALUE(elem) (elem.second)
#define DICT_INSERT(dict, key, value) dict[key] = value;
#endif

// Run inference
std::unique_ptr<NeuropodValueMap> TorchNeuropodBackend::infer(const NeuropodValueMap &inputs)
{
    torch::NoGradGuard guard;

    // Get inference schema
    const auto &method    = model_->get_method("forward");
    const auto &schema    = SCHEMA(method);
    const auto &arguments = schema.arguments();

    // Whether or not this model expects a dictionary as an input
    bool is_dict_input = false;

    // Torch 1.2.0 adds a ClassType argument to every model
    bool has_class_type = false;

#if CAFFE2_NIGHTLY_VERSION >= 20190717
    if (arguments.size() > 0 && arguments.at(0).type()->kind() == c10::TypeKind::ClassType)
    {
        has_class_type = true;
    }
#endif

    if (arguments.size() == 2 && has_class_type && arguments.at(1).type()->kind() == c10::TypeKind::DictType)
    {
        is_dict_input = true;
    }

    if (arguments.size() == 1 && arguments.at(0).type()->kind() == c10::TypeKind::DictType)
    {
        is_dict_input = true;
    }

    // Define the vector of inputs and add the inputs
    std::vector<torch::jit::IValue> torch_inputs(arguments.size() - (has_class_type ? 1 : 0));
    if (is_dict_input)
    {
        // This model expects a dict as input
        MAKE_DICT(input_dict);
        for (const auto &entry : inputs)
        {
            const auto  device = get_torch_device(input_device_mapping_.at(entry.first));
            const auto &value  = get_ivalue_from_torch_tensor(entry.second);

            DICT_INSERT(input_dict, entry.first, maybe_set_device(value, device));
        }

        torch_inputs.at(0) = input_dict;
    }
    else
    {
        // Pass inputs normally
        for (const auto &entry : inputs)
        {
            const auto  input_name = entry.first;
            const auto &input_data = get_ivalue_from_torch_tensor(entry.second);

            const auto arg_index = schema.argumentIndexWithName(input_name);
            if (!arg_index.has_value())
            {
                NEUROPOD_ERROR("Input '" << input_name.c_str() << "' does not exist. Model inputs " << schema);
            }

            const auto device = get_torch_device(input_device_mapping_.at(input_name));

            torch_inputs.at(arg_index.value() - (has_class_type ? 1 : 0)) = maybe_set_device(input_data, device);
        }
    }

    // Run inference
    c10::IValue result = model_->forward(torch_inputs);

    // Get outputs
    auto to_return = stdx::make_unique<NeuropodValueMap>();

    if (result.isGenericDict())
    {
        const auto &outputs_dict = ELEMENTS(result.toGenericDict());
        for (const auto &elem : outputs_dict)
        {
            // Get the name of the tensor
            const std::string &name = KEY(elem).toString()->string();
            // Todo include tensor_type if available
            insert_value_in_output(*to_return, name, VALUE(elem));
        }
    }
    else if (result.isTensor() || result.isGenericList())
    {
        if (output_specs_.empty())
        {
            NEUROPOD_ERROR("Model did not return dict and output spec is empty");
        }
        if (output_specs_.size() != 1)
        {
            NEUROPOD_ERROR("Model did not return dict and output spec is not size 1");
        }

        auto &name        = output_specs_[0].name;
        auto &tensor_type = output_specs_[0].type;
        insert_value_in_output(*to_return, name, result, true, tensor_type);
    }
    else
    {
        NEUROPOD_ERROR("Torchscript model output type not supported in neuropod");
    }

    return to_return;
}

REGISTER_NEUROPOD_BACKEND(TorchNeuropodBackend, "torchscript")

} // namespace neuropods
