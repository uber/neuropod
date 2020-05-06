//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropod/backends/neuropod_backend.hh"
#include "neuropod/backends/tensorflow/tf_tensor.hh"
#include "neuropod/neuropod.hh"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorflow
{

// Forward declare tensorflow::Session and tensorflow::Tensor
class Session;
class Tensor;

} // namespace tensorflow

namespace neuropod
{

// This backend can execute TensorFlow models
class TensorflowNeuropodBackend : public NeuropodBackendWithDefaultAllocator<TensorflowNeuropodTensor>
{
private:
    std::unique_ptr<tensorflow::Session> session_;

    // Cached access to callable handles
    std::unordered_map<std::string, int64_t> callable_handle_cache_;

    // Map from a neuropod node name to the appropriate node in the TF graph
    std::unordered_map<std::string, std::string> node_name_mapping_;

    // The outputs of the model. This is from the neuropod model config
    std::vector<std::string> output_names_;

    // Get a callable given feeds and fetches
    // This will try to use a cached one if possible
    int64_t get_callable(const std::map<std::string, tensorflow::Tensor> &tensor_feeds,
                         const std::map<std::string, std::string> &       tensor_fetches);

public:
    TensorflowNeuropodBackend(const std::string &neuropod_path, const RuntimeOptions &options);

    ~TensorflowNeuropodBackend();

protected:
    // Run inference with a set of requested outputs
    std::unique_ptr<NeuropodValueMap> infer_internal(const NeuropodValueMap &        inputs,
                                                     const std::vector<std::string> &requested_outputs);

    // A method that loads the underlying model
    void load_model_internal();
};

} // namespace neuropod
