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

#include "neuropod/backends/neuropod_backend.hh"

#include "fmt/ranges.h"
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
        const NeuropodDeviceType target_device = item.second;
        NeuropodDevice           device        = Device::CPU;
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

std::shared_ptr<NeuropodValue> Sealer::seal(const std::string &name, const std::shared_ptr<NeuropodValue> &value)
{
    auto device_it = device_mapping_.find(name);
    if (device_it != device_mapping_.end())
    {
        return value->as_tensor()->to(device_it->second);
    }

    NEUROPOD_ERROR("Tried to seal a tensor with name '{}', but could not find it in the spec", name);
}

NeuropodValueMap Sealer::seal(const NeuropodValueMap &inputs)
{
    NeuropodValueMap out;
    for (const auto &item : inputs)
    {
        out[item.first] = seal(item.first, item.second);
    }

    return out;
}

void validate_tensors_against_specs(const NeuropodValueMap &       tensors,
                                    const std::vector<TensorSpec> &specs,
                                    const std::string &            debug_spec_name)
{
    // A vector of all the tensor names in the spec.
    // This is used to check for extra tensors that were provided,
    // but not in the spec
    std::unordered_set<std::string> spec_tensor_names;

    // All instances of a symbol in a specification must
    // resolve to the same value at runtime. See below for more detail
    std::unordered_map<std::string, int64_t> symbol_actual_map;

    for (const auto &spec : specs)
    {
        // Add the item's name to a set used to check for extra tensors
        spec_tensor_names.emplace(spec.name);

        // Try to find a tensor with the same name as this item
        auto tensor_it = tensors.find(spec.name);
        if (tensor_it == tensors.end())
        {
            // For now, all tensors are optional
            // TODO(vip): Fix this once we have a better way of marking items as optional
            continue;
        }

        // Get the tensor
        const auto &tensor = tensor_it->second->as_tensor();

        // Validate data type
        if (tensor->get_tensor_type() != spec.type)
        {
            // Throw an error
            NEUROPOD_ERROR("Tensor '{}' in the {} is expected to be of type {}, but was of type {}",
                           spec.name,
                           debug_spec_name,
                           spec.type,
                           tensor->get_tensor_type());
        }

        // Validate the number of dimensions
        if (tensor->get_dims().size() != spec.dims.size())
        {
            // Throw an error
            NEUROPOD_ERROR("Tensor '{}' in the {} is expected to have {} dimensions, but had {}",
                           spec.name,
                           debug_spec_name,
                           spec.dims.size(),
                           tensor->get_dims().size());
        }

        // Validate the shape
        for (size_t i = 0; i < spec.dims.size(); i++)
        {
            auto dim      = tensor->get_dims()[i];
            auto expected = spec.dims[i];

            if (expected.value == -1)
            {
                // Any value of dim is okay
                continue;
            }
            else if (expected.value > 0) // NOLINT(readability-else-after-return)
            {
                // Check that we have the expected number of items
                if (dim != expected.value)
                {
                    // Throw an error
                    NEUROPOD_ERROR("Dim {} of tensor '{}' in the {} is expected to be of size {}, but was of size {}",
                                   i,
                                   spec.name,
                                   debug_spec_name,
                                   expected.value,
                                   dim);
                }
            }
            else if (expected.value < -1)
            {
                // `expected` is a symbol.
                // Every instance of `expected` should have the same value
                // For example, if a symbol of "num_classes" is used multiple times in the spec,
                // all instances must have the same value
                auto actual_it = symbol_actual_map.find(expected.symbol);
                if (actual_it != symbol_actual_map.end())
                {
                    // We've seen this symbol before
                    auto actual_value = actual_it->second;

                    // Make sure this usage matches the previous value
                    if (dim != actual_value)
                    {
                        // Throw an error
                        NEUROPOD_ERROR(
                            "All dims with symbol '{}' should be the same size. "
                            "Dim {} of tensor '{}' in the {} was expected to be of size {}, but was of size {}",
                            expected.symbol,
                            i,
                            spec.name,
                            debug_spec_name,
                            actual_value,
                            dim);
                    }
                }
                else
                {
                    // This is the first time we're seeing this symbol
                    // Add it to the map so we can check future occurrances of this symbol
                    symbol_actual_map[expected.symbol] = dim;
                }
            }
            else
            {
                // Throw an error
                NEUROPOD_ERROR(
                    "Invalid value of expected shape for item in the {}: {}", debug_spec_name, expected.value);
            }
        }
    }

    // Check for extra tensors that are not included in the spec
    std::vector<std::string> unexpected_tensors;
    for (const auto &item : tensors)
    {
        if (spec_tensor_names.find(item.first) == spec_tensor_names.end())
        {
            unexpected_tensors.emplace_back(item.first);
        }
    }

    if (!unexpected_tensors.empty())
    {
        // Throw an error
        NEUROPOD_ERROR("Tensor name(s) '{}' are not found in the {}", unexpected_tensors, debug_spec_name);
    }
}

NeuropodBackend::~NeuropodBackend() = default;

NeuropodBackend::NeuropodBackend(const std::string &neuropod_path, RuntimeOptions options)
    : model_config_(load_model_config(neuropod_path)),
      neuropod_path_(neuropod_path),
      options_(std::move(options)),
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
    return model_config_->inputs;
}

const std::vector<TensorSpec> &NeuropodBackend::get_outputs() const
{
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

    if (!options_.disable_shape_and_type_checking)
    {
        // Validate inputs
        validate_tensors_against_specs(inputs, get_inputs(), "input spec");
    }

    // Seal the inputs
    auto sealed = sealer_->seal(inputs);

    // Run inference
    auto out = infer_internal(sealed, requested_outputs);

    if (!options_.disable_shape_and_type_checking)
    {
        // Validate outputs
        validate_tensors_against_specs(*out, get_outputs(), "output spec");
    }

    return out;
}

std::unique_ptr<NeuropodValueMap> NeuropodBackend::infer_internal(const NeuropodValueMap &        inputs,
                                                                  const std::vector<std::string> &requested_outputs)
{
    // We're not doing any filtering
    if (requested_outputs.empty())
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

std::unique_ptr<NeuropodValueMap> NeuropodBackend::infer_internal(const NeuropodValueMap & /*unused*/)
{
    NEUROPOD_ERROR("Backend implementations must provide a `infer_internal` implementation");
}

} // namespace neuropod
