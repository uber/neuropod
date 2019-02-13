//
// Uber, Inc. (c) 2018
//

#include "neuropods.hh"

#include "neuropods/internal/backend_registration.hh"
#include "neuropods/internal/config_utils.hh"
#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/internal/tensor_store.hh"

namespace neuropods
{

struct Neuropod::impl
{
    std::shared_ptr<NeuropodBackend> backend;
    std::unique_ptr<ModelConfig> model_config;
};

Neuropod::Neuropod(const std::string &neuropod_path) : pimpl(stdx::make_unique<Neuropod::impl>())
{
    // Find the right backend to use and load the neuropod
    pimpl->model_config = load_model_config(neuropod_path);
    pimpl->backend      = get_backend_for_type(pimpl->model_config->platform)(neuropod_path, pimpl->model_config);
}

Neuropod::Neuropod(const std::string &neuropod_path, const std::string &backend_name)
    : pimpl(stdx::make_unique<Neuropod::impl>())
{
    // Load the neuropod using the specified backend
    pimpl->model_config = load_model_config(neuropod_path);
    pimpl->backend      = get_backend_by_name(backend_name)(neuropod_path, pimpl->model_config);
}

Neuropod::Neuropod(const std::string &neuropod_path, std::shared_ptr<NeuropodBackend> backend)
    : pimpl(stdx::make_unique<Neuropod::impl>())
{
    // Load the model config and use the backend that was provided by the user
    pimpl->model_config = load_model_config(neuropod_path);
    pimpl->backend = std::move(backend);
}

Neuropod::~Neuropod() = default;

std::unique_ptr<NeuropodInputBuilder> Neuropod::get_input_builder()
{
    return stdx::make_unique<NeuropodInputBuilder>(pimpl->backend);
}

std::unique_ptr<NeuropodOutputData> Neuropod::infer(const std::unique_ptr<TensorStore> &inputs)
{
    // Run inference
    auto output_tensor_store = pimpl->backend->infer(*inputs);

    // Wrap in a NeuropodOutputData so users can easily access the data
    return stdx::make_unique<NeuropodOutputData>(std::move(output_tensor_store));
}

const std::vector<TensorSpec> &Neuropod::get_inputs() const
{
    return pimpl->model_config->inputs;
}

const std::vector<TensorSpec> &Neuropod::get_outputs() const
{
    return pimpl->model_config->outputs;
}

} // namespace neuropods
