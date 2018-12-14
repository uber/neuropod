//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <vector>

#include "neuropods/internal/tensor_types.hh"

namespace neuropods
{

struct TensorSpec
{
    TensorSpec(const std::string &name, const std::vector<int64_t> dims, const TensorType type);
    ~TensorSpec();

    const std::string          name;
    const std::vector<int64_t> dims;
    const TensorType           type;
};

struct ModelConfig
{
    ModelConfig(const std::string &            name,
                const std::string &            platform,
                const std::vector<TensorSpec> &inputs,
                const std::vector<TensorSpec> &outputs);
    ~ModelConfig();

    const std::string name;
    const std::string platform;

    const std::vector<TensorSpec> inputs;
    const std::vector<TensorSpec> outputs;
};

ModelConfig load_model_config(const std::string &neuropod_path);
ModelConfig load_model_config(std::istream &input_stream);

} // namespace neuropods
