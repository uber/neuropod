//
// Uber, Inc. (c) 2018
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "neuropods/internal/memory_utils.hh"
#include "neuropods/internal/tensor_types.hh"

namespace neuropods
{

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
};

std::unique_ptr<ModelConfig> load_model_config(const std::string &neuropod_path);
std::unique_ptr<ModelConfig> load_model_config(std::istream &input_stream);

} // namespace neuropods
