//
// Uber, Inc. (c) 2019
//

#include "infer_utils.hh"

#include "neuropods/backends/torchscript/type_utils.hh"
#include "neuropods/backends/torchscript/version_utils.hh"
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

void promote_torch_to_global()
{
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
}

std::shared_ptr<torch::jit::script::Module> load_model_from_path(const std::string &             graph_path,
                                                                 const std::vector<std::string> &custom_op_paths,
                                                                 const torch::Device &           device)
{
    // This is so custom ops work correctly
    promote_torch_to_global();

    // Load custom ops
    // TODO(vip): Add a flag allowing users to opt out of loading custom ops
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

} // namespace

TorchInferenceWrapper::TorchInferenceWrapper(const std::string &             torchscript_model_path,
                                             const std::vector<std::string> &custom_op_paths,
                                             const torch::Device &           device)
    : model_(load_model_from_path(torchscript_model_path, custom_op_paths, device))
{
}

TorchInferenceWrapper::~TorchInferenceWrapper() = default;

// Run inference
torch::jit::IValue TorchInferenceWrapper::infer(const std::unordered_map<std::string, torch::jit::IValue> &inputs, bool remap_dict_input)
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
    if (remap_dict_input && is_dict_input)
    {
        // This model expects a dict as input
        MAKE_DICT(input_dict);
        for (const auto &entry : inputs)
        {
            DICT_INSERT(input_dict, entry.first, entry.second);
        }

        torch_inputs.at(0) = input_dict;
    }
    else
    {
        // Pass inputs normally
        for (const auto &entry : inputs)
        {
            const auto  input_name = entry.first;

            const auto arg_index = schema.argumentIndexWithName(input_name);
            if (!arg_index.has_value())
            {
                NEUROPOD_ERROR("Input '" << input_name.c_str() << "' does not exist. Model inputs " << schema);
            }

            torch_inputs.at(arg_index.value() - (has_class_type ? 1 : 0)) = entry.second;
        }
    }

    // Run inference
    return model_->forward(torch_inputs);
}

} // namespace neuropods
