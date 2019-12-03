//
// Uber, Inc. (c) 2019
//

#include "neuropods/backends/tensorflow/tf_backend.hh"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session.h"

#include <json/json.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dlfcn.h>

namespace neuropods
{
namespace
{

// Load a mapping from neuropod node names to node names in the TF graph
void setup_node_mapping_and_init_ops(std::istream &                                ifs,
                                     std::unordered_map<std::string, std::string> &mapping,
                                     std::vector<std::string> &                    init_ops)
{
    // Parse it
    Json::CharReaderBuilder rbuilder;
    Json::Value             obj;

    std::string parse_err;
    bool        parsingSuccessful = Json::parseFromStream(rbuilder, ifs, &obj, &parse_err);

    if (!parsingSuccessful)
    {
        NEUROPOD_ERROR("Error parsing TF Neuropod Config JSON: " + parse_err);
    }

    // Make sure that node_name_mapping exists and is an object
    if (!obj["node_name_mapping"].isObject())
    {
        NEUROPOD_ERROR("'node_name_mapping' must be an object in the tf neuropod config");
    }

    const auto node_name_mapping = obj["node_name_mapping"];
    const auto node_names        = node_name_mapping.getMemberNames();
    for (const std::string &node_name : node_names)
    {
        const auto val = node_name_mapping[node_name];
        if (!val.isString())
        {
            NEUROPOD_ERROR("All values in 'node_name_mapping' in the tf neuropod config must be strings");
        }

        mapping[node_name] = val.asString();
    }

    // Get the init ops (if any)
    if (obj.isMember("init_op_names"))
    {
        const auto &ops = obj["init_op_names"];
        for (Json::Value::ArrayIndex i = 0; i != ops.size(); i++)
        {
            init_ops.push_back(ops[i].asString());
        }
    }
}

} // namespace

TensorflowNeuropodBackend::TensorflowNeuropodBackend(const std::string &           neuropod_path,
                                                     std::unique_ptr<ModelConfig> &model_config,
                                                     const RuntimeOptions &        options)
    : NeuropodBackendWithDefaultAllocator<TensorflowNeuropodTensor>(neuropod_path), session_(tensorflow::NewSession({}))
{
#ifndef __APPLE__
    // We need to do this so the custom ops can see the symbols from TF
    void *libtensorflow = dlopen("libtensorflow_framework.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (libtensorflow == nullptr)
    {
        NEUROPOD_ERROR("Failed to promote libtensorflow to RTLD_GLOBAL. Error from dlopen: " << dlerror());
    }
#endif

    // Load custom ops (if any)
    for (const auto &item : model_config->custom_ops)
    {
        if (dlopen(loader_->get_file_path("0/ops/" + item).c_str(), RTLD_NOW) == nullptr)
        {
            NEUROPOD_ERROR("Failed to load custom op. Error from dlopen: " << dlerror());
        }
    }

    // Get a stream for the graph
    auto graph_stream = loader_->get_istream_for_file("0/data/model.pb");

    // Create a buffer of the right size
    graph_stream->seekg(0, graph_stream->end);
    std::streamsize   graph_length = graph_stream->tellg();
    std::vector<char> buffer(graph_length);

    // Read into the buffer
    graph_stream->seekg(0, graph_stream->beg);
    graph_stream->read(buffer.data(), graph_length);
    if (graph_stream->fail())
    {
        NEUROPOD_ERROR("Error reading TensorFlow GraphDef for neuropod " << neuropod_path);
    }

    // Read the GraphDef
    tensorflow::GraphDef graph;
    tensorflow::ParseProtoUnlimited(&graph, buffer.data(), buffer.size());

    // Create a session
    auto status = session_->Create(graph);
    if (!status.ok())
    {
        NEUROPOD_ERROR("Error loading TensorFlow graph: " << status.error_message());
    }

    // Setup the nodename mapping and get the init ops (if any)
    std::vector<std::string> init_ops;
    auto                     config_stream = loader_->get_istream_for_file("0/config.json");
    setup_node_mapping_and_init_ops(*config_stream, node_name_mapping_, init_ops);

    // Get a list of the output nodes
    for (const auto &output : model_config->outputs)
    {
        output_names_.emplace_back(output.name);
    }

    for (const auto &op_name : init_ops)
    {
        auto status = session_->Run({}, {}, {op_name}, nullptr);
        if (!status.ok())
        {
            NEUROPOD_ERROR("Error running TensorFlow init op: " << op_name);
        }
    }
}

TensorflowNeuropodBackend::~TensorflowNeuropodBackend() = default;

std::unique_ptr<NeuropodValueMap> TensorflowNeuropodBackend::infer(const NeuropodValueMap &inputs)
{
    return infer(inputs, {});
}

// Run inference with a set of requested outputs
std::unique_ptr<NeuropodValueMap> TensorflowNeuropodBackend::infer(const NeuropodValueMap &        inputs,
                                                                   const std::vector<std::string> &requested_outputs)
{
    // Get the set of outputs we want to compute
    const auto &output_names = requested_outputs.size() > 0 ? requested_outputs : output_names_;

    // Transform neuropod output names to node names in the graph
    std::vector<std::string> output_node_names;
    for (const auto &name : output_names)
    {
        const auto node_name = node_name_mapping_.find(name);
        if (node_name == node_name_mapping_.end())
        {
            NEUROPOD_ERROR("Node " << name
                                   << " not found in node_name_mapping. "
                                      "Ensure that all items in the input/output spec have a corresponding item "
                                      "in the node_name_mapping.");
        }

        output_node_names.emplace_back(node_name->second);
    }

    // Loop through all the input tensors and setup the inputs
    std::vector<std::pair<std::string, tensorflow::Tensor>> tf_inputs;
    for (const auto &entry : inputs)
    {
        const auto node_name = node_name_mapping_.find(entry.first);
        if (node_name == node_name_mapping_.end())
        {
            NEUROPOD_ERROR("Node " << entry.first
                                   << " not found in node_name_mapping. "
                                      "Ensure that all items in the input/output spec have a corresponding item "
                                      "in the node_name_mapping.");
        }

        const auto &input_data =
            std::dynamic_pointer_cast<NativeDataContainer<tensorflow::Tensor &>>(entry.second)->get_native_data();

        tf_inputs.emplace_back(std::make_pair(node_name->second, input_data));
    }

    std::vector<tensorflow::Tensor> outputs;
    auto                            status = session_->Run(tf_inputs, output_node_names, {}, &outputs);
    if (!status.ok())
    {
        NEUROPOD_ERROR("TensorFlow error: " << status.error_message())
    }

    // Read the outputs
    auto to_return = stdx::make_unique<NeuropodValueMap>();
    for (size_t i = 0; i < output_names.size(); ++i)
    {
        const auto &output_name   = output_names[i];
        auto &      output_tensor = outputs[i];
        const auto  tensor_type   = get_neuropod_type_from_tf_type(output_tensor.dtype());
        (*to_return)[output_name] = make_tensor<TensorflowNeuropodTensor>(tensor_type, std::move(output_tensor));
    }

    return to_return;
}

REGISTER_NEUROPOD_BACKEND(TensorflowNeuropodBackend, "tensorflow")

} // namespace neuropods
