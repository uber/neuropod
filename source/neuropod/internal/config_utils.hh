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

#pragma once

#include "neuropod/internal/memory_utils.hh"
#include "neuropod/internal/tensor_types.hh"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuropod
{

// Device types that are supported in the Neuropod configuration
typedef int NeuropodDeviceType;
namespace DeviceType
{
constexpr int CPU = 0;
constexpr int GPU = 1;
}; // namespace DeviceType

// A struct that stores the value of a dimension
struct Dimension
{
    Dimension(int64_t value);
    Dimension(std::string symbol);

    bool operator==(const Dimension &other) const;

    // The value
    // -1 == Any value is allowed (None/null)
    // -2 == Symbol
    int64_t value;

    // The name of this symbol (if it is a symbol)
    std::string symbol;
};

// A struct that stores a specification for a tensor
struct TensorSpec
{
    TensorSpec(std::string name, std::vector<Dimension> dims, const TensorType type);

    const std::string            name;
    const std::vector<Dimension> dims;
    const TensorType             type;
};

// A struct that stores the expected inputs and outputs of a model
struct ModelConfig
{
    const std::string name;
    const std::string platform;

    // The requested versions of the platform specified as a semver range
    // e.g. `1.13.1` or `> 1.13.1`
    // See the following URLs for examples and more info:
    // - https://semver.org/
    // - https://docs.npmjs.com/misc/semver#ranges
    // - https://docs.npmjs.com/misc/semver#advanced-range-syntax
    const std::string platform_version_semver;

    const std::vector<TensorSpec> inputs;
    const std::vector<TensorSpec> outputs;

    const std::vector<std::string> custom_ops;

    // A map from an input tensor name to a device type
    const std::unordered_map<std::string, NeuropodDeviceType> input_tensor_device;
};

std::unique_ptr<ModelConfig> load_model_config(const std::string &neuropod_path);
std::unique_ptr<ModelConfig> load_model_config(std::istream &input_stream);

} // namespace neuropod
