//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/backends/tensorflow/tf_tensor.hh"

#include <string>
#include <unordered_map>
#include <vector>

namespace tensorflow
{

// Forward declare tensorflow::Session
class Session;

} // namespace tensorflow

namespace neuropods
{

// This backend can execute TensorFlow models
class TensorflowNeuropodBackend : public NeuropodBackendWithDefaultAllocator<TensorflowNeuropodTensor>
{
private:
    std::unique_ptr<tensorflow::Session> session_;

    // Map from a neuropod node name to the appropriate node in the TF graph
    std::unordered_map<std::string, std::string> node_name_mapping_;

    // The outputs of the model. This is from the neuropod model config
    std::vector<std::string> output_names_;

public:
    explicit TensorflowNeuropodBackend(const std::string &           neuropod_path,
                                       std::unique_ptr<ModelConfig> &model_config,
                                       const RuntimeOptions &        options);

    ~TensorflowNeuropodBackend();

    // Run inference
    std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &inputs);

    // Run inference with a set of requested outputs
    std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &        inputs,
                                            const std::vector<std::string> &requested_outputs);
};

} // namespace neuropods
