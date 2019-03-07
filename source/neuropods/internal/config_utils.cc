//
// Uber, Inc. (c) 2018
//

#include "config_utils.hh"

#include <fstream>
#include <jsoncpp/json/json.h>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace neuropods
{

namespace
{

// Get graph path from a neuropod path
std::string get_config_path(const std::string &neuropod_path)
{
    if (neuropod_path.back() == '/')
    {
        return neuropod_path + "config.json";
    }

    return neuropod_path + "/config.json";
}

[[noreturn]] void throw_neuropod_config_error(const std::string &err) {
    std::stringstream ss;
    ss << "------------------------------" << std::endl;
    ss << "Error loading neuropod config!" << std::endl;
    ss << err << std::endl;
    ss << "Please check your config file" << std::endl;
    ss << "------------------------------" << std::endl;
    throw std::runtime_error(ss.str());
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

std::vector<int64_t> get_dims_from_json(const Json::Value &json_shape)
{
    // Make sure that the shape is an array
    if (json_shape.isArray())
    {
        std::vector<int64_t> out;
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
                // TODO(vip): implement
            }
            else
            {
                throw_neuropod_config_error(
                    "All items in 'shape' must be either null, a string, or a positive integer.");
            }
        }

        return out;
    }
    else
    {
        throw_neuropod_config_error("'shape' must be an array. Please check your config file");
    }
}

} // namespace

TensorSpec::TensorSpec(const std::string &name, const std::vector<int64_t> dims, const TensorType type)
    : name(name), dims(dims), type(type)
{
}


TensorSpec::~TensorSpec() = default;

ModelConfig::ModelConfig(const std::string &            name,
                         const std::string &            platform,
                         const std::vector<TensorSpec> &inputs,
                         const std::vector<TensorSpec> &outputs)
    : name(name), platform(platform), inputs(inputs), outputs(outputs)
{
}

ModelConfig::~ModelConfig() = default;

std::unique_ptr<ModelConfig> load_model_config(const std::string &neuropod_path)
{
    auto path = get_config_path(neuropod_path);

    // Load the config file
    std::ifstream ifs(path);

    if (!ifs)
    {
        std::stringstream ss;
        ss << "Error loading config file '";
        ss << path;
        ss << "'!";
        throw std::runtime_error(ss.str());
    }

    return load_model_config(ifs);
}

std::unique_ptr<ModelConfig> load_model_config(std::istream &input_stream)
{
    // Parse it
    Json::Reader reader;
    Json::Value  obj;
    reader.parse(input_stream, obj);

    // Make sure that name and platform are strings
    if (!obj["name"].isString() || !obj["platform"].isString())
    {
        throw_neuropod_config_error("'name' and 'platform' must be strings.");
    }

    const std::string  name        = obj["name"].asString();
    const std::string  platform    = obj["platform"].asString();
    const Json::Value &input_spec  = obj["input_spec"];
    const Json::Value &output_spec = obj["output_spec"];

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

    return stdx::make_unique<ModelConfig>(name, platform, inputs, outputs);
}

} // namespace neuropods
