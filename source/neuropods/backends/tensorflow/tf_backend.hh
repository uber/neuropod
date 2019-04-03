//
// Uber, Inc. (c) 2018
//

#pragma once

#include <unordered_map>
#include <string>
#include <vector>

#include <tensorflow/c/c_api.h>

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/backends/tensorflow/tf_tensor.hh"
#include "neuropods/backends/tensorflow/tf_wrappers.hh"

namespace neuropods
{


// This backend can execute TensorFlow models
class TensorflowNeuropodBackend : public NeuropodBackendWithDefaultAllocator<TensorflowNeuropodTensor>
{
private:
    // Setup setup inputs given a TensorStore
    void setup_inputs(const TensorStore &       inputs,
                      std::vector<TF_Output> &  input_ops,
                      std::vector<TF_Tensor *> &input_values);


    // Check the status of the last TF API call and throw an exception if
    // there was an error
    void check_status() const;

    // Loads a frozen graph from a protobuf file
    void load_graph(const std::string &graph_path);

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
    explicit TensorflowNeuropodBackend(const std::string &neuropod_path, std::unique_ptr<ModelConfig> model_config);

    ~TensorflowNeuropodBackend();

    // Run inference
    std::unique_ptr<TensorStore> infer(const std::unordered_set<std::shared_ptr<NeuropodTensor>> &inputs);
};

} // namespace neuropods
