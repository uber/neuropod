//
// Uber, Inc. (c) 2018
//

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

// A struct that stores a specification for a tensor
struct TensorSpec
{
    TensorSpec(const std::string &name, const std::vector<int64_t> dims, const TensorType type);
    ~TensorSpec();

    const std::string          name;
    const std::vector<int64_t> dims;
    const TensorType           type;
};

// A struct that stores the expected inputs and outputs of a model
struct ModelConfig
{
    const std::string name;
    const std::string platform;

    const std::vector<TensorSpec> inputs;
    const std::vector<TensorSpec> outputs;

    const std::vector<std::string> custom_ops;

    // A map from an input tensor name to a device type
    const std::unordered_map<std::string, NeuropodDeviceType> input_tensor_device;
};

std::unique_ptr<ModelConfig> load_model_config(const std::string &neuropod_path);
std::unique_ptr<ModelConfig> load_model_config(std::istream &input_stream);

} // namespace neuropod
