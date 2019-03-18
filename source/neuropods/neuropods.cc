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

Neuropod::Neuropod(const std::string &neuropod_path)
    : Neuropod(neuropod_path, std::unordered_map<std::string, std::string>())
{
}

// Find the right backend to use and load the neuropod
Neuropod::Neuropod(const std::string &neuropod_path, const std::unordered_map<std::string, std::string> &default_backend_overrides)
    : model_config_(load_model_config(neuropod_path)),
      backend_(get_backend_for_type(default_backend_overrides, model_config_->platform)(neuropod_path, model_config_))
{
}

// Load the neuropod using the specified backend
Neuropod::Neuropod(const std::string &neuropod_path, const std::string &backend_name)
    : model_config_(load_model_config(neuropod_path)),
      backend_(get_backend_by_name(backend_name)(neuropod_path, model_config_))
{
}

// Load the model config and use the backend that was provided by the user
Neuropod::Neuropod(const std::string &neuropod_path, std::shared_ptr<NeuropodBackend> backend)
    : model_config_(load_model_config(neuropod_path)),
      backend_(backend)
{
}

Neuropod::~Neuropod() = default;

std::unique_ptr<NeuropodInputBuilder> Neuropod::get_input_builder()
{
    return stdx::make_unique<NeuropodInputBuilder>(backend_);
}

std::unique_ptr<TensorStore> Neuropod::infer(const std::unique_ptr<TensorStore> &inputs)
{
    // Run inference
    return backend_->infer(*inputs);
}

const std::vector<TensorSpec> &Neuropod::get_inputs() const
{
    return model_config_->inputs;
}

const std::vector<TensorSpec> &Neuropod::get_outputs() const
{
    return model_config_->outputs;
}

} // namespace neuropods
