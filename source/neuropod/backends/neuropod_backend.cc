//
// Uber, Inc. (c) 2019
//

#include "neuropod/backends/neuropod_backend.hh"

#include "fmt/ranges.h"
#include "neuropod/internal/config_utils.hh"
#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/neuropod_loader.hh"

namespace neuropod
{

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
        for (int i = 0; i < spec.dims.size(); i++)
        {
            auto dim      = tensor->get_dims()[i];
            auto expected = spec.dims[i];

            if (expected.value == -1)
            {
                // Any value of dim is okay
                continue;
            }
            else if (expected.value > 0)
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

    if (unexpected_tensors.size() > 0)
    {
        // Throw an error
        NEUROPOD_ERROR("Tensor name(s) '{}' are not found in the {}", unexpected_tensors, debug_spec_name);
    }
}

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

    // Validate inputs
    validate_tensors_against_specs(inputs, get_inputs(), "input spec");

    // Run inference
    auto out = infer_internal(inputs, requested_outputs);

    // Validate outputs
    validate_tensors_against_specs(*out, get_outputs(), "output spec");

    return out;
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
