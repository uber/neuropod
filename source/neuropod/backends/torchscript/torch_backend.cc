//
// Uber, Inc. (c) 2018
//

#include "torch_backend.hh"

#include "neuropod/backends/torchscript/type_utils.hh"
#include "neuropod/internal/tensor_types.hh"

#include <caffe2/core/macros.h>

#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <dlfcn.h>

namespace neuropod
{

namespace
{

std::shared_ptr<torch::jit::script::Module> load_model_from_path(std::istream &                  graph_stream,
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
        NEUROPOD_ERROR("Failed to promote libtorch to RTLD_GLOBAL. Error from dlopen: {}", dlerror());
    }
#endif

    for (const auto &path : custom_op_paths)
    {
        if (dlopen(path.c_str(), RTLD_NOW) == nullptr)
        {
            NEUROPOD_ERROR("Failed to load custom op. Error from dlopen: {}", dlerror());
        }
    }

#if CAFFE2_NIGHTLY_VERSION >= 20190717
    auto model = std::make_shared<torch::jit::script::Module>(torch::jit::load(graph_stream, device));
#else
    auto model = torch::jit::load(graph_stream, device);
#endif
    return model;
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
            NEUROPOD_ERROR("Neuropod got a list of type '{}' for tensor '{}'."
                           "Only tensors or lists of strings are supported",
                           list[0].tagKind(),
                           name);
        }
    }
    else
    {
        NEUROPOD_ERROR("Neuropod returned an invalid type! All outputs must be tensors"
                       "or lists of strings. Got type '{}' for tensor '{}'",
                       value.tagKind(),
                       name);
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

// Used to avoid loading the same custom op multiple times
std::unordered_set<std::string> loaded_op_hashes;
std::mutex                      loaded_op_mutex;

} // namespace

TorchNeuropodBackend::TorchNeuropodBackend(const std::string &neuropod_path, const RuntimeOptions &options)
    : NeuropodBackendWithDefaultAllocator<TorchNeuropodTensor>(neuropod_path),
      options_(options),
      input_device_mapping_(model_config_->input_tensor_device)
{
    // Get the model from the neuropod
    auto graph_stream = loader_->get_istream_for_file("0/data/model.pt");

    // Custom ops
    // Make sure we don't load a custom op twice
    std::vector<std::string> custom_ops;
    for (const auto &item : model_config_->custom_ops)
    {
        const auto path = "0/ops/" + item;
        const auto hash = loader_->get_hash_for_file(path);

        // Don't load a custom op if we've already loaded it
        std::lock_guard<std::mutex> lock(loaded_op_mutex);
        if (loaded_op_hashes.count(hash) == 0)
        {
            custom_ops.emplace_back(loader_->get_file_path(path));
            loaded_op_hashes.insert(hash);
        }
    }

    model_ = load_model_from_path(*graph_stream,
                                  custom_ops,

                                  // Load the model onto the appropriate device (ideally a GPU if we have one available)
                                  // Note: this uses the options set in the initializer list above
                                  get_torch_device(DeviceType::GPU));

    if (!model_)
    {
        NEUROPOD_ERROR("Failed to load TorchScript graph for neuropod {}", neuropod_path);
    }

    for (const auto &tensor_spec : model_config_->outputs)
    {
        output_specs_.emplace_back(tensor_spec);
    }
}

TorchNeuropodBackend::~TorchNeuropodBackend() = default;

torch::Device TorchNeuropodBackend::get_torch_device(NeuropodDeviceType target_device)
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

#if CAFFE2_NIGHTLY_VERSION >= 20200115
#define MAKE_DICT(name, type) torch::Dict<std::string, type> name;
#elif CAFFE2_NIGHTLY_VERSION >= 20191010
#define MAKE_DICT(name, type) c10::impl::GenericDict name(c10::AnyType::get(), c10::AnyType::get());
#elif CAFFE2_NIGHTLY_VERSION >= 20190717
#define MAKE_DICT(name, type) c10::impl::GenericDict name((c10::impl::deprecatedUntypedDict()));
#elif CAFFE2_NIGHTLY_VERSION >= 20190601
#define MAKE_DICT(name, type) auto name = c10::make_dict<torch::jit::IValue, torch::jit::IValue>();
#else
#define MAKE_DICT(name, type) torch::ivalue::UnorderedMap name;
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
std::unique_ptr<NeuropodValueMap> TorchNeuropodBackend::infer_internal(const NeuropodValueMap &inputs)
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
        MAKE_DICT(tensor_input_dict, torch::Tensor);
        MAKE_DICT(str_input_dict, torch::List<std::string>);

        for (const auto &entry : inputs)
        {
            const auto  device = get_torch_device(input_device_mapping_.at(entry.first));
            const auto &value  = get_ivalue_from_torch_tensor(entry.second);

            if (value.isTensor())
            {
                // .to(device) is a no-op if the tensor is already transferred
                DICT_INSERT(tensor_input_dict, entry.first, value.toTensor().to(device));
            }
            else
            {
#if CAFFE2_NIGHTLY_VERSION >= 20190717
                DICT_INSERT(str_input_dict, entry.first, c10::impl::toTypedList<std::string>(value.toGenericList()));
#else
                DICT_INSERT(str_input_dict, entry.first, value);
#endif
            }
        }

        // TODO(vip): This assumes a model only takes in string "tensors" or tensors, but not both
        // Refactor to add support for both and add documentation
        if (*arguments.at(has_class_type ? 1 : 0).type()->cast<torch::DictType>()->getValueType() ==
            *torch::TensorType::get())
        {
            torch_inputs.at(0) = tensor_input_dict;
        }
        else
        {
            torch_inputs.at(0) = str_input_dict;
        }
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
                NEUROPOD_ERROR("Input '{}' does not exist. Model inputs {}", input_name, schema);
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

} // namespace neuropod
