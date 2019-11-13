//
// Uber, Inc. (c) 2019
//

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/internal/neuropod_loader.hh"


namespace neuropods
{

NeuropodBackend::NeuropodBackend() = default;
NeuropodBackend::~NeuropodBackend() = default;

NeuropodBackend::NeuropodBackend(const std::string &neuropod_path)
{
    loader_ = get_loader(neuropod_path);
}

std::unique_ptr<NeuropodValueMap> NeuropodBackend::infer(const NeuropodValueMap &inputs, const std::vector<std::string> &requested_outputs)
{
    // We're not doing any filtering
    if (requested_outputs.size() == 0)
    {
        return infer(inputs);
    }

    // Run inference and get all the outputs
    auto data = infer(inputs);
    auto out = stdx::make_unique<NeuropodValueMap>();

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

} // namespace neuropods
