//
// Uber, Inc. (c) 2019
//

#include "neuropod/backends/neuropod_backend.hh"

#include "neuropod/internal/config_utils.hh"
#include "neuropod/internal/neuropod_loader.hh"

namespace neuropod
{

NeuropodBackend::NeuropodBackend()  = default;
NeuropodBackend::~NeuropodBackend() = default;

NeuropodBackend::NeuropodBackend(const std::string &neuropod_path)
    : model_config_(load_model_config(neuropod_path)), neuropod_path_(neuropod_path)
{
    loader_ = get_loader(neuropod_path);
}

void NeuropodBackend::load_model()
{
    if (!is_model_loaded_)
    {
        load_model_internal();
        is_model_loaded_ = true;
    }
    else
    {
        NEUROPOD_ERROR(
            "The model has already been loaded. This usually means that "
            "`load_model_at_construction` was set to true (default) or `load_model()` was already explicitly called");
    }
}

const std::vector<TensorSpec> &NeuropodBackend::get_inputs() const
{
    if (model_config_ == nullptr)
    {
        static const std::vector<TensorSpec> empty = {};
        return empty;
    }

    return model_config_->inputs;
}

const std::vector<TensorSpec> &NeuropodBackend::get_outputs() const
{
    if (model_config_ == nullptr)
    {
        static const std::vector<TensorSpec> empty = {};
        return empty;
    }

    return model_config_->outputs;
}

const std::string &NeuropodBackend::get_name() const
{
    return model_config_->name;
}

const std::string &NeuropodBackend::get_platform() const
{
    return model_config_->platform;
}

std::unique_ptr<NeuropodValueMap> NeuropodBackend::infer(const NeuropodValueMap &        inputs,
                                                         const std::vector<std::string> &requested_outputs)
{
    // Make sure the model is loaded
    if (!is_model_loaded_)
    {
        NEUROPOD_ERROR("The model was not loaded before calling `infer`. This usually means that "
                       "`load_model_at_construction` was set to false and `load_model()` was not explicitly called");
    }

    // Run inference
    return infer_internal(inputs, requested_outputs);
}

std::unique_ptr<NeuropodValueMap> NeuropodBackend::infer_internal(const NeuropodValueMap &        inputs,
                                                                  const std::vector<std::string> &requested_outputs)
{
    // We're not doing any filtering
    if (requested_outputs.size() == 0)
    {
        return infer_internal(inputs);
    }

    // Run inference and get all the outputs
    auto data = infer_internal(inputs);
    auto out  = stdx::make_unique<NeuropodValueMap>();

    // Filter to the requested outputs
    for (const auto &tensor_name : requested_outputs)
    {
        auto tensor = data->find(tensor_name);
        if (tensor == data->end())
        {
            NEUROPOD_ERROR("Tried to request a tensor that does not exist: {}", tensor_name);
        }

        (*out)[tensor_name] = std::move(tensor->second);
    }

    return out;
}

std::unique_ptr<NeuropodValueMap> NeuropodBackend::infer_internal(const NeuropodValueMap &inputs)
{
    NEUROPOD_ERROR("Backend implementations must provide a `infer_internal` implementation");
}

} // namespace neuropod
