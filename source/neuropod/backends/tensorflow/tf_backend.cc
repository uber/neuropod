//
// Uber, Inc. (c) 2019
//

#include "neuropod/backends/tensorflow/tf_backend.hh"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session.h"

#include <json/json.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <dlfcn.h>

namespace neuropod
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

// Throws an error if `status` is not ok
void check_tf_status(const tensorflow::Status &status)
{
    if (!status.ok())
    {
        NEUROPOD_ERROR("TensorFlow error: " << status.error_message())
    }
}

// Get TF session options given Neuropod RuntimeOptions
tensorflow::SessionOptions get_tf_opts(const RuntimeOptions & /*unused*/)
{
    tensorflow::SessionOptions opts;

    // Don't preallocate the entire GPU
    auto gpu_opts = opts.config.mutable_gpu_options();
    gpu_opts->set_allow_growth(true);
    return opts;
}

// Used to avoid loading the same custom op multiple times
std::unordered_set<std::string> loaded_op_hashes;
std::mutex                      loaded_op_mutex;

// Note: this is intentionally not using references here because we're sorting the vectors
std::string get_handle_cache_key(std::vector<std::string> tensor_feeds, std::vector<std::string> tensor_fetches)
{
    // TODO(vip): Do an unsorted check before doing a sorted one
    // Check if we have an existing callable for our set of inputs and outputs
    std::sort(tensor_feeds.begin(), tensor_feeds.end());
    std::sort(tensor_fetches.begin(), tensor_fetches.end());

    std::string cache_key;
    for (const auto &piece : tensor_feeds)
    {
        cache_key += piece + ",";
    }

    cache_key += "->";

    for (const auto &piece : tensor_fetches)
    {
        cache_key += piece + ",";
    }

    return cache_key;
}

} // namespace

TensorflowNeuropodBackend::TensorflowNeuropodBackend(const std::string &           neuropod_path,
                                                     std::unique_ptr<ModelConfig> &model_config,
                                                     const RuntimeOptions &        options)
    : NeuropodBackendWithDefaultAllocator<TensorflowNeuropodTensor>(neuropod_path),
      session_(tensorflow::NewSession(get_tf_opts(options)))
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
        const auto path = "0/ops/" + item;
        const auto hash = loader_->get_hash_for_file(path);

        // Don't load a custom op if we've already loaded it
        std::lock_guard<std::mutex> lock(loaded_op_mutex);
        if (loaded_op_hashes.count(hash) == 0)
        {
            if (dlopen(loader_->get_file_path(path).c_str(), RTLD_NOW) == nullptr)
            {
                NEUROPOD_ERROR("Failed to load custom op. Error from dlopen: " << dlerror());
            }

            loaded_op_hashes.insert(hash);
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
        check_tf_status(session_->Run({}, {}, {op_name}, nullptr));
    }
}

TensorflowNeuropodBackend::~TensorflowNeuropodBackend()
{
    // Release all the callables we cached
    for (const auto &item : callable_handle_cache_)
    {
        check_tf_status(session_->ReleaseCallable(item.second));
    }
}

int64_t TensorflowNeuropodBackend::get_callable(const std::vector<std::string> &tensor_feeds,
                                                const std::vector<std::string> &tensor_fetches)
{
    tensorflow::Session::CallableHandle handle;

    const auto cache_key     = get_handle_cache_key(tensor_feeds, tensor_fetches);
    auto       cached_handle = callable_handle_cache_.find(cache_key);
    if (cached_handle != callable_handle_cache_.end())
    {
        // Cache hit!
        handle = cached_handle->second;
    }
    else
    {
        // Cache miss...
        SPDLOG_DEBUG("TF: Callable cache miss. Creating new callable...");

        // Used for setting the inputs and outputs of the subgraph we want to run
        tensorflow::CallableOptions opts;

        for (const auto &item : tensor_feeds)
        {
            opts.add_feed(item);

            // TODO(vip): Once we explicitly control devices, do something like this:
            // opts.mutable_feed_devices()->insert({item, device_name});
        }

        for (const auto &item : tensor_fetches)
        {
            opts.add_fetch(item);
        }

        // Make the callable using the options we set above
        // Note: this callable will be released in the destructor
        check_tf_status(session_->MakeCallable(opts, &handle));

        // Add it to our cache
        callable_handle_cache_[cache_key] = handle;
    }

    return handle;
}

std::unique_ptr<NeuropodValueMap> TensorflowNeuropodBackend::infer(const NeuropodValueMap &inputs)
{
    return infer(inputs, {});
}

// Run inference with a set of requested outputs
std::unique_ptr<NeuropodValueMap> TensorflowNeuropodBackend::infer(const NeuropodValueMap &        inputs,
                                                                   const std::vector<std::string> &requested_outputs)
{
    // In TensorFlow, a callable is a way of running a subgraph given a set of inputs and
    // outputs. It's very similar to `session_->Run` except it has support for more fine-grained
    // control over tensor devices. See https://github.com/tensorflow/tensorflow/issues/5902
    // for more details.

    // Fetches and feeds for our callable
    std::vector<std::string> tensor_fetches;
    std::vector<std::string> tensor_feeds;

    // Get the set of outputs we want to compute
    const auto &output_names = requested_outputs.size() > 0 ? requested_outputs : output_names_;

    // Transform neuropod output names to node names in the graph
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

        // Add this node name as an output of the subgraph we want to run
        tensor_fetches.emplace_back(node_name->second);
    }

    // Loop through all the input tensors and setup the inputs
    std::vector<tensorflow::Tensor> tf_inputs;
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

        // Get the TensorFlow tensor from the Neuropod tensor
        const auto &input_data =
            std::dynamic_pointer_cast<NativeDataContainer<tensorflow::Tensor &>>(entry.second)->get_native_data();

        // Add this node name as an input to the subgraph we want to run
        tensor_feeds.emplace_back(node_name->second);

        // Add the tensor to our vector of inputs
        tf_inputs.emplace_back(input_data);
    }

    // Create a callable handle and a vector to store our outputs
    tensorflow::Session::CallableHandle handle = get_callable(tensor_feeds, tensor_fetches);
    std::vector<tensorflow::Tensor>     outputs;

    // Run the callable with the vector of inputs we created above
    check_tf_status(session_->RunCallable(handle, tf_inputs, &outputs, nullptr));

    // Read the outputs and wrap them in `NeuropodTensor`s
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

} // namespace neuropod
