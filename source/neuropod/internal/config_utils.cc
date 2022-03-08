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

#include "config_utils.hh"

#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/neuropod_loader.hh"

#include <json/json.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <cpp-semver.hpp>

namespace neuropod
{

namespace
{

[[noreturn]] void throw_neuropod_config_error(const std::string &message)
{
    NEUROPOD_ERROR("Error loading neuropod config! Please check your config file. {}", message);
}

const std::unordered_map<std::string, TensorType> type_mapping = {
    {"float32", FLOAT_TENSOR},
    {"float64", DOUBLE_TENSOR},
    {"string", STRING_TENSOR},

    {"int8", INT8_TENSOR},
    {"int16", INT16_TENSOR},
    {"int32", INT32_TENSOR},
    {"int64", INT64_TENSOR},

    {"uint8", UINT8_TENSOR},
    {"uint16", UINT16_TENSOR},
    {"uint32", UINT32_TENSOR},
    {"uint64", UINT64_TENSOR},
};

// Convert a string data type to a TensorType
TensorType convert_to_tensor_type(const Json::Value &dtype)
{
    if (!dtype.isString())
    {
        throw_neuropod_config_error("'dtype' must be a string.");
    }

    const auto dt  = dtype.asString();
    auto       got = type_mapping.find(dt);
    if (got == type_mapping.end())
    {
        std::stringstream ss;
        ss << "The specified data type '";
        ss << dt;
        ss << "' is invalid.";
        throw_neuropod_config_error(ss.str());
    }
    else
    {
        return got->second;
    }
}

std::vector<Dimension> get_dims_from_json(const Json::Value &json_shape)
{
    // Make sure that the shape is an array
    if (json_shape.isArray())
    {
        // The dims to return
        std::vector<Dimension> out;
        for (const auto &item : json_shape)
        {
            // Get the number and do some validation
            // Each item in the array must be either null or a positive integer
            if (item.isInt() && item.asInt64() > 0)
            {
                out.emplace_back(item.asInt64());
            }
            else if (item.isNull())
            {
                // A dim of size -1 means we won't check the size of that dim
                out.emplace_back(-1);
            }
            else if (item.isString())
            {
                // It is a symbol
                out.emplace_back(item.asString());
            }
            else
            {
                throw_neuropod_config_error(
                    "All items in 'shape' must be either null, a string, or a positive integer.");
            }
        }

        return out;
    }

    throw_neuropod_config_error("'shape' must be an array. Please check your config file");
}

} // namespace

Dimension::Dimension(int64_t v) : value(v) {}
Dimension::Dimension(std::string s) : value(-2), symbol(std::move(s)) {}

bool Dimension::operator==(const Dimension &other) const
{
    if (value == other.value)
    {
        // If it's a symbol, make sure the symbol name matches
        return value == -2 ? symbol == other.symbol : true;
    }

    return false;
}

TensorSpec::TensorSpec(std::string n, std::vector<Dimension> d, const TensorType t)
    : name(std::move(n)), dims(std::move(d)), type(t)
{
}

std::unique_ptr<ModelConfig> load_model_config(const std::string &neuropod_path)
{
    auto loader = get_loader(neuropod_path);

    // Load the config file
    auto stream = loader->get_istream_for_file("config.json");
    if (!stream)
    {
        NEUROPOD_ERROR("Error loading config file for neuropod '{}'", neuropod_path);
    }

    return load_model_config(*stream);
}

std::unique_ptr<ModelConfig> load_model_config(std::istream &input_stream)
{
    // Parse it
    Json::CharReaderBuilder rbuilder;
    Json::Value             obj;

    std::string parse_err;
    bool        parsingSuccessful = Json::parseFromStream(rbuilder, input_stream, &obj, &parse_err);

    if (!parsingSuccessful)
    {
        throw_neuropod_config_error("Error parsing JSON: " + parse_err);
    }

    // Make sure that name and platform are strings
    if (!obj["name"].isString() || !obj["platform"].isString())
    {
        throw_neuropod_config_error("'name' and 'platform' must be strings.");
    }

    const std::string  name        = obj["name"].asString();
    const std::string  platform    = obj["platform"].asString();
    const Json::Value &input_spec  = obj["input_spec"];
    const Json::Value &output_spec = obj["output_spec"];

    // By default, any version is okay
    std::string platform_version_semver = "*";
    if (obj.isMember("platform_version_semver"))
    {
        platform_version_semver = obj["platform_version_semver"].asString();
    }

    // Make sure that it's a valid semver version or range
    if (!semver::valid(platform_version_semver))
    {
        throw_neuropod_config_error("The provided platform version (" + platform_version_semver +
                                    ") was not a valid semver version or range. See https://semver.org/ and "
                                    "https://docs.npmjs.com/misc/semver#ranges");
    }

    // Get the inputs
    std::vector<TensorSpec> inputs;
    for (const auto &spec : input_spec)
    {
        // Make sure name is a string
        if (!spec["name"].isString())
        {
            throw_neuropod_config_error("'name' must be a string.");
        }

        inputs.emplace_back(
            spec["name"].asString(), get_dims_from_json(spec["shape"]), convert_to_tensor_type(spec["dtype"]));
    }

    // Get the outputs
    std::vector<TensorSpec> outputs;
    for (const auto &spec : output_spec)
    {
        // Make sure name is a string
        if (!spec["name"].isString())
        {
            throw_neuropod_config_error("'name' must be a string.");
        }

        outputs.emplace_back(
            spec["name"].asString(), get_dims_from_json(spec["shape"]), convert_to_tensor_type(spec["dtype"]));
    }

    // Get the list of custom ops if any
    std::vector<std::string> custom_ops;
    if (obj.isMember("custom_ops"))
    {
        const Json::Value &items = obj["custom_ops"];
        for (const auto &item : items)
        {
            custom_ops.emplace_back(item.asString());
        }
    }

    // Load the device mapping if any
    std::unordered_map<std::string, NeuropodDeviceType> input_tensor_device;
    if (obj.isMember("input_tensor_device"))
    {
        const Json::Value &device_mapping = obj["input_tensor_device"];
        const auto         names          = device_mapping.getMemberNames();
        for (const auto &tensor_name : names)
        {
            const auto type = device_mapping[tensor_name].asString();
            if (type == "GPU")
            {
                input_tensor_device[tensor_name] = DeviceType::GPU;
            }
            else if (type == "CPU")
            {
                input_tensor_device[tensor_name] = DeviceType::CPU;
            }
            else
            {
                throw_neuropod_config_error("Invalid device type: " + type);
            }
        }
    }
    else
    {
        // Default all the tensors to GPU
        // TODO(vip): Remove this on version increase
        for (const auto &input : inputs)
        {
            input_tensor_device[input.name] = DeviceType::GPU;
        }
    }

    // Not directly using make_unique because of brace initialization
    return stdx::make_unique<ModelConfig>(
        ModelConfig{name, platform, platform_version_semver, inputs, outputs, custom_ops, input_tensor_device});
}

} // namespace neuropod
