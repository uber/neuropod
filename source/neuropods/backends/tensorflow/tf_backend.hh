//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/backends/tensorflow/tf_tensor.hh"
#include "neuropods/backends/tensorflow/tf_wrappers.hh"

#include <tensorflow/c/c_api.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace neuropods
{

// This backend can execute TensorFlow models
class TensorflowNeuropodBackend : public NeuropodBackendWithDefaultAllocator<TensorflowNeuropodTensor>
{
private:
    // Check the status of the last TF API call and throw an exception if
    // there was an error
    void check_status() const;

    // Loads a frozen graph from a protobuf file
    bool load_graph(std::istream &graph_stream);

    // Run target ops in the graph
    void run_target_ops(const std::vector<std::string> &target_op_names);

    // Pointer to the status of a TF call
    TF_StatusPtr status_;
    // Pointer to the TF graph
    TF_GraphPtr graph_;
    // Pointer to TF session options
    TF_SessionOptionsPtr session_opts_;
    // Pointer to the TF session
    TF_SessionPtr session_;

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
    std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &inputs, const std::vector<std::string> &requested_outputs);
};

} // namespace neuropods
