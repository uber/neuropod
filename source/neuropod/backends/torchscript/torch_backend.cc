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

#if CAFFE2_NIGHTLY_VERSION >= 20200115
#define MAKE_DICT(name, type) torch::Dict<std::string, type> name
#elif CAFFE2_NIGHTLY_VERSION >= 20191010
#define MAKE_DICT(name, type) c10::impl::GenericDict name(c10::AnyType::get(), c10::AnyType::get())
#elif CAFFE2_NIGHTLY_VERSION >= 20190717
#define MAKE_DICT(name, type) c10::impl::GenericDict name((c10::impl::deprecatedUntypedDict()))
#elif CAFFE2_NIGHTLY_VERSION >= 20190601
#define MAKE_DICT(name, type) auto name = c10::make_dict<torch::jit::IValue, torch::jit::IValue>()
#else
#define MAKE_DICT(name, type) torch::ivalue::UnorderedMap name
#endif

#if CAFFE2_NIGHTLY_VERSION >= 20190717
#define SCHEMA(method) method.function().getSchema()
#else
#define SCHEMA(method) method.getSchema()
#endif

#if CAFFE2_NIGHTLY_VERSION >= 20190601
#define KEY(elem) (elem.key())
#define VALUE(elem) (elem.value())
#define DICT_INSERT(dict, key, value) dict.insert(key, value)
#else
#define KEY(elem) (elem.first)
#define VALUE(elem) (elem.second)
#define DICT_INSERT(dict, key, value) dict[key] = value
#endif

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
        const auto err = dlerror();
        if (err == nullptr)
        {
            NEUROPOD_ERROR("Failed to promote libtorch to RTLD_GLOBAL; this likely means the neuropod backend library "
                           "was not built correctly");
        }
        else
        {
            NEUROPOD_ERROR("Failed to promote libtorch to RTLD_GLOBAL. Error from dlopen: {}", err);
        }
    }
#endif

    for (const auto &path : custom_op_paths)
    {
        if (dlopen(path.c_str(), RTLD_NOW) == nullptr)
        {
            const auto err = dlerror();
            if (err == nullptr)
            {
                NEUROPOD_ERROR("Failed to load custom op. dlopen failed but no error was available");
            }
            else
            {
                NEUROPOD_ERROR("Failed to load custom op. Error from dlopen: {}", err);
            }
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
        // .contiguous() is a no-op if the tensor is already contiguous
        auto tensor = value.toTensor().to(torch::kCPU).contiguous();

        // Get the type and make a TorchNeuropodTensor
        auto neuropod_tensor_type = get_neuropod_type_from_torch_type(tensor.scalar_type());
        auto neuropod_tensor      = make_tensor<TorchNeuropodTensor>(neuropod_tensor_type, tensor);

        // Add it to our output
        auto &to_set = output[name];
        if (!to_set)
        {
            to_set = std::move(neuropod_tensor);
        }
        else
        {
            NEUROPOD_ERROR("An item with name `{}` was already returned by this model. Please ensure your model does "
                           "not have duplicate outputs",
                           name);
        }
    }
#if CAFFE2_NIGHTLY_VERSION >= 20200421
    else if (value.isList())
#else
    else if (value.isGenericList())
#endif
    {
        // A list of strings
        // This is used in place of string tensors because torch does not
        // have native support for string tensors
        auto &tensor = value;

#if CAFFE2_NIGHTLY_VERSION >= 20200421
        const auto &list = tensor.toListRef();
#else
        const auto &list = tensor.toGenericListRef();
#endif

        // if tensor_type string or no tensor_type and empty list or list containing actual string
        if ((has_type && tensor_type == TensorType::STRING_TENSOR) || (!has_type && list.empty()) ||
            (!has_type && list[0].isString()))
        {
            // Make a TorchNeuropodTensor
            auto neuropod_tensor = stdx::make_unique<TorchNeuropodTensor<std::string>>(tensor);

            // Add it to our output
            auto &to_set = output[name];
            if (!to_set)
            {
                to_set = std::move(neuropod_tensor);
            }
            else
            {
                NEUROPOD_ERROR("An item with name `{}` was already returned by this model. Please ensure your model "
                               "does not have duplicate outputs",
                               name);
            }
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

// Insert all the elements of a dict into a NeuropodValueMap
void process_dict(NeuropodValueMap &output, const c10::IValue &item)
{
    const auto &dict = ELEMENTS(item.toGenericDict());
    for (const auto &elem : dict)
    {
        // Get the name of the tensor
        const std::string &name = KEY(elem).toString()->string();
        insert_value_in_output(output, name, VALUE(elem));
    }
}

// Used to avoid loading the same custom op multiple times
std::unordered_set<std::string> loaded_op_hashes;
std::mutex                      loaded_op_mutex;

} // namespace

TorchNeuropodBackend::TorchNeuropodBackend(const std::string &neuropod_path, const RuntimeOptions &options)
    : NeuropodBackendWithDefaultAllocator<TorchNeuropodTensor>(neuropod_path, options)
{
    if (options.load_model_at_construction)
    {
        load_model();
    }
}

void TorchNeuropodBackend::load_model_internal()
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
        NEUROPOD_ERROR("Failed to load TorchScript graph for neuropod {}", neuropod_path_);
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

    return torch::Device(torch::kCUDA, static_cast<char>(options_.visible_device));
}

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

    // Torch 1.10.2 adds UnionType support in TorchScript
    bool dict_value_is_union_type = false;

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

#if CAFFE2_NIGHTLY_VERSION >= 20220127
    if (is_dict_input && arguments.at(has_class_type ? 1 : 0).type()->cast<torch::DictType>()->getValueType()->kind() ==
                             c10::TypeKind::UnionType)
    {
        dict_value_is_union_type = true;
    }
#endif

    // Define the vector of inputs and add the inputs
    std::vector<torch::jit::IValue> torch_inputs(arguments.size() - (has_class_type ? 1 : 0));
    if (is_dict_input && !dict_value_is_union_type)
    {
        // This model expects a dict as input
        MAKE_DICT(tensor_input_dict, torch::Tensor);
        MAKE_DICT(str_input_dict, torch::List<std::string>);

        for (const auto &entry : inputs)
        {
            const auto &value = get_ivalue_from_torch_tensor(entry.second);

            if (value.isTensor())
            {
                DICT_INSERT(tensor_input_dict, entry.first, value.toTensor());
            }
            else
            {
#if CAFFE2_NIGHTLY_VERSION >= 20200421
                DICT_INSERT(str_input_dict, entry.first, c10::impl::toTypedList<std::string>(value.toList()));
#elif CAFFE2_NIGHTLY_VERSION >= 20190717
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
#if CAFFE2_NIGHTLY_VERSION >= 20220127
    // In Torch 1.10.2, TorchScript introduced UnionType and it now supports Dict[str, Union[List[str], torch.Tensor]]
    // as model input type. We would like to support this input type in neuropod torchscript backend
    else if (is_dict_input && dict_value_is_union_type)
    {
        const auto &value_type_ptr = torch::UnionType::create({torch::ListType::ofStrings(), torch::TensorType::get()});
        c10::impl::GenericDict input_dict(torch::StringType::get(), value_type_ptr);

        for (const auto &entry : inputs)
        {
            const auto &value = get_ivalue_from_torch_tensor(entry.second);
            input_dict.insert(entry.first, value);
        }
        torch_inputs.at(0) = input_dict;
    }
#endif
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
                NEUROPOD_ERROR(
                    "An tensor named '{}' was provided, but does not exist in the input schema of the "
                    "TorchScript model. Please ensure your model expects an input with that name. Schema: {}",
                    input_name,
                    schema);
            }

            torch_inputs.at(static_cast<size_t>(arg_index.value() - (has_class_type ? 1 : 0))) = input_data;
        }
    }

    // Run inference
    c10::IValue result = model_->forward(torch_inputs);

    // Get outputs
    auto to_return = stdx::make_unique<NeuropodValueMap>();

    if (result.isGenericDict())
    {
        process_dict(*to_return, result);
    }
#if CAFFE2_NIGHTLY_VERSION >= 20200421
    else if (result.isTensor() || result.isList())
#else
    else if (result.isTensor() || result.isGenericList())
#endif
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
    else if (result.isTuple())
    {
        auto  tuple = result.toTuple();
        auto &elems = tuple->elements();

        // Macros to handle namedtuples (Torch >= 1.3.0)
#if CAFFE2_NIGHTLY_VERSION >= 20191010
        const auto tuple_type     = result.type()->cast<torch::TupleType>();
        const bool is_named_tuple = tuple_type && tuple_type->schema();
#define GET_NAME(i) tuple_type->schema()->arguments()[i].name()
#else
        const bool is_named_tuple = false;
#define GET_NAME(i) ""
#endif
        if (is_named_tuple)
        {
            // This is a named tuple
            // NOLINTNEXTLINE(modernize-loop-convert): Can't always use a range based loop here
            for (size_t i = 0; i < elems.size(); i++)
            {
                insert_value_in_output(*to_return, GET_NAME(i), elems.at(i));
            }
        }
        else
        {
            // Each item in this tuple should be a dict
            for (const auto &item : elems)
            {
                if (item.isGenericDict())
                {
                    process_dict(*to_return, item);
                }
                else
                {
                    NEUROPOD_ERROR("When returning a tuple, each item must be a dict. Got {}", item.tagKind());
                }
            }
        }

#undef GET_NAME
    }
    else { NEUROPOD_ERROR("Torchscript model output type not supported in neuropod"); }

    return to_return;
}

REGISTER_NEUROPOD_BACKEND(TorchNeuropodBackend, "torchscript", STR(TORCH_VERSION))

} // namespace neuropod
