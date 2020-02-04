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

NeuropodBackend::NeuropodBackend(const std::string &neuropod_path) : model_config_(load_model_config(neuropod_path))
{
    loader_ = get_loader(neuropod_path);
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

std::unique_ptr<NeuropodValueMap> NeuropodBackend::infer(const NeuropodValueMap &        inputs,
                                                         const std::vector<std::string> &requested_outputs)
{
    // We're not doing any filtering
    if (requested_outputs.size() == 0)
    {
        return infer(inputs);
    }

    // Run inference and get all the outputs
    auto data = infer(inputs);
    auto out  = stdx::make_unique<NeuropodValueMap>();

    // Filter to the requested outputs
    for (const auto &tensor_name : requested_outputs)
    {
        auto tensor = data->find(tensor_name);
        if (tensor == data->end())
        {
            NEUROPOD_ERROR("Tried to request a tensor that does not exist: " << tensor_name);
        }

        (*out)[tensor_name] = std::move(tensor->second);
    }

    return out;
}

} // namespace neuropod
