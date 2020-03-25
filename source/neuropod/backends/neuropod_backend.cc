//
// Uber, Inc. (c) 2019
//

#include "neuropod/backends/neuropod_backend.hh"

#include "neuropod/internal/config_utils.hh"
#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/neuropod_loader.hh"

namespace neuropod
{

namespace
{

std::unordered_map<std::string, NeuropodDevice> get_device_mapping(const ModelConfig &   model_config,
                                                                   const RuntimeOptions &options)
{
    // Generate the device mapping
    std::unordered_map<std::string, NeuropodDevice> device_mapping;
    for (const auto &item : model_config.input_tensor_device)
    {
        NeuropodDeviceType target_device = item.second;
        NeuropodDevice     device        = Device::CPU;
        if (target_device == DeviceType::GPU && options.visible_device != Device::CPU)
        {
            device = options.visible_device;
        }

        device_mapping[item.first] = device;
    }

    return device_mapping;
}

} // namespace

Sealer::Sealer(std::unordered_map<std::string, NeuropodDevice> device_mapping)
    : device_mapping_(std::move(device_mapping))
{
}
Sealer::~Sealer() = default;

std::shared_ptr<NeuropodValue> Sealer::seal(const std::string &name, const std::shared_ptr<NeuropodValue> &value) const
{
    // TODO: do this better
    if (dynamic_cast<SealedNeuropodTensor *>(value.get()) != nullptr)
    {
        // This is already a sealed tensor
        // Note: this'll currently run multiple times on tensors that don't return sealed values, but those
        // should effectively be noops
        return value;
    }

    auto device_it = device_mapping_.find(name);
    if (device_it != device_mapping_.end())
    {
        return value->as_tensor()->seal(device_it->second);
    }
    else
    {
        NEUROPOD_ERROR("Tried to seal a tensor with name '{}', but could not find it in the spec", name);
    }
}

void Sealer::seal(NeuropodValueMap &items, const std::string &name, const std::shared_ptr<NeuropodValue> &value) const
{
    items[name] = seal(name, value);
}

NeuropodValueMap Sealer::seal(const NeuropodValueMap &inputs) const
{
    NeuropodValueMap out;
    for (auto &item : inputs)
    {
        seal(out, item.first, item.second);
    }

    return out;
}

NeuropodBackend::NeuropodBackend()  = default;
NeuropodBackend::~NeuropodBackend() = default;

NeuropodBackend::NeuropodBackend(const std::string &neuropod_path, const RuntimeOptions &options)
    : model_config_(load_model_config(neuropod_path)),
      neuropod_path_(neuropod_path),
      options_(options),
      sealer_(stdx::make_unique<Sealer>(get_device_mapping(*model_config_, options_)))
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

    // Seal the inputs
    auto sealed = sealer_->seal(inputs);

    // Run inference
    return infer_internal(sealed, requested_outputs);
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

Sealer NeuropodBackend::get_sealer()
{
    if (!sealer_)
    {
        NEUROPOD_ERROR("Tried to get a Sealer, but it was null. This usually means you tried getting a sealer for the "
                       "test backend")
    }

    return *sealer_;
}

} // namespace neuropod
