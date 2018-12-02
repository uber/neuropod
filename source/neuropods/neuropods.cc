//
// Uber, Inc. (c) 2018
//

#include "neuropods.hh"

#include "neuropods/internal/backend_registration.hh"
#include "neuropods/internal/neuropod_input_data.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/internal/tensor_store.hh"
#include "neuropods/proxy/neuropod_proxy.hh"

namespace neuropods
{

struct Neuropod::impl
{
    std::shared_ptr<NeuropodBackend> backend;
};

Neuropod::Neuropod(const std::string &neuropod_path) : pimpl(std::make_unique<Neuropod::impl>())
{
    // TODO(vip): construct the right backend based on the neuropod at the specified
    // path.
    // For now, run everything with the python backend
    pimpl->backend = get_backend_for_type("python")(neuropod_path);
}

Neuropod::Neuropod(std::shared_ptr<NeuropodProxy> backend_proxy) : pimpl(std::make_unique<Neuropod::impl>())
{
    pimpl->backend = std::move(backend_proxy);
}

Neuropod::~Neuropod() = default;

std::unique_ptr<NeuropodInputBuilder> Neuropod::get_input_builder()
{
    return std::make_unique<NeuropodInputBuilder>(pimpl->backend);
}

std::unique_ptr<NeuropodOutputData> Neuropod::infer(
    const std::unique_ptr<NeuropodInputData, NeuropodInputDataDeleter> &inputs)
{
    // Run inference
    auto output_tensor_store = pimpl->backend->infer(*inputs->data);

    // Wrap in a NeuropodOutputData so users can easily access the data
    return std::make_unique<NeuropodOutputData>(std::move(output_tensor_store));
}

} // namespace neuropods
