/* Copyright (c) 2020 The Neuropod Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "neuropod/backends/tensorflow/tf_backend.hh"

#include "neuropod/backends/tensorflow/saved_model/loader.h"
#include "neuropod/backends/tensorflow/tf_utils.hh"
#include "neuropod/neuropod.hh"
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
        NEUROPOD_ERROR("Error parsing TF Neuropod Config JSON: {}", parse_err);
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

// Get TF session options given Neuropod RuntimeOptions
tensorflow::SessionOptions get_tf_opts(const RuntimeOptions & /*unused*/)
{
    tensorflow::SessionOptions opts;

    // Don't preallocate the entire GPU
    auto gpu_opts = opts.config.mutable_gpu_options();
    gpu_opts->set_allow_growth(true);
    opts.config.set_allow_soft_placement(true);
    opts.config.set_log_device_placement(false);

    // Note: we can't use GPUOptions::visible_device_list as it is a per process setting
    //
    // From: https://github.com/tensorflow/tensorflow/issues/18861#issuecomment-385610497
    // Unfortunately, though `visible_deivces_list` is included in `ConfigProto`, it is
    // actually a per-process setting. In fact, this is true of almost all options inside
    // the GPUOptions protocol buffer.

    // We set the device of the graph in the constructor below by moving each node to the
    // target device before creating a session.
    return opts;
}

// Used to avoid loading the same custom op multiple times
std::unordered_set<std::string> loaded_op_hashes;
std::mutex                      loaded_op_mutex;

std::string get_handle_cache_key(const std::map<std::string, tensorflow::Tensor> &tensor_feeds,
                                 const std::map<std::string, std::string> &       tensor_fetches)
{
    std::string cache_key;
    for (const auto &item : tensor_feeds)
    {
        // item.first is the node name in the TF graph
        cache_key += item.first + ",";
    }

    cache_key += "->";

    for (const auto &item : tensor_fetches)
    {
        // item.first is the node name in the TF graph
        cache_key += item.first + ",";
    }

    return cache_key;
}

} // namespace

TensorflowNeuropodBackend::TensorflowNeuropodBackend(const std::string &neuropod_path, const RuntimeOptions &options)
    : NeuropodBackendWithDefaultAllocator<TensorflowNeuropodTensor>(neuropod_path, options),
      session_(tensorflow::NewSession(get_tf_opts(options)))
{
    if (options.load_model_at_construction)
    {
        load_model();
    }
}

void TensorflowNeuropodBackend::load_saved_model()
{
    tensorflow::SavedModelBundle bundle;
    constexpr char               kSavedModelTagServe[] = "serve";
    check_tf_status(tensorflow::LoadSavedModel(
        get_tf_opts(options_), {}, loader_->get_file_path("0/data/savedmodel"), {kSavedModelTagServe}, &bundle));

    session_ = std::move(bundle.session);

    // Get the input and output node names for the `serving_default` signature in the savedmodel
    // See https://www.tensorflow.org/guide/saved_model#specifying_signatures_during_export
    // for more details
    const auto &signature_def = bundle.GetSignatures().at("serving_default");

    for (const auto &item : signature_def.inputs())
    {
        node_name_mapping_[item.first] = item.second.name();
    }

    for (const auto &item : signature_def.outputs())
    {
        node_name_mapping_[item.first] = item.second.name();
    }
}

void TensorflowNeuropodBackend::load_frozen_graph(std::istream &graph_stream)
{
    // Create a buffer of the right size
    graph_stream.seekg(0, graph_stream.end);
    auto              graph_length = static_cast<size_t>(graph_stream.tellg());
    std::vector<char> buffer(graph_length);

    // Read into the buffer
    graph_stream.seekg(0, graph_stream.beg);
    graph_stream.read(buffer.data(), static_cast<std::streamsize>(graph_length));
    if (graph_stream.fail())
    {
        NEUROPOD_ERROR("Error reading TensorFlow GraphDef for neuropod {}", neuropod_path_);
    }

    // Read the GraphDef
    tensorflow::GraphDef graph;
    tensorflow::ParseProtoUnlimited(&graph, buffer.data(), buffer.size());

    // Move the graph to the target device
    move_graph_to_device(graph, *session_, options_.visible_device);

    // Create a session
    auto status = session_->Create(graph);
    if (!status.ok())
    {
        NEUROPOD_ERROR("Error loading TensorFlow graph: {}", status.error_message());
    }

    // Setup the nodename mapping and get the init ops (if any)
    std::vector<std::string> init_ops;
    auto                     config_stream = loader_->get_istream_for_file("0/config.json");
    setup_node_mapping_and_init_ops(*config_stream, node_name_mapping_, init_ops);

    // Run init ops if any
    for (const auto &op_name : init_ops)
    {
        check_tf_status(session_->Run({}, {}, {op_name}, nullptr));
    }
}

void TensorflowNeuropodBackend::load_model_internal()
{
    // Load custom ops (if any)
    for (const auto &item : model_config_->custom_ops)
    {
        const auto path = "0/ops/" + item;
        const auto hash = loader_->get_hash_for_file(path);

        // Don't load a custom op if we've already loaded it
        std::lock_guard<std::mutex> lock(loaded_op_mutex);
        if (loaded_op_hashes.count(hash) == 0)
        {
            if (dlopen(loader_->get_file_path(path).c_str(), RTLD_NOW) == nullptr)
            {
                const auto err = dlerror();
                if (err == nullptr)
                {
                    NEUROPOD_ERROR("Failed to load custom op. dlopen failed but no error was available");
                }
                else
                {
                    NEUROPOD_ERROR("Failed to load custom op. Error from dlopen: {}", err);
                }
            }

            loaded_op_hashes.insert(hash);
        }
    }

    // Get a list of the output nodes
    for (const auto &output : model_config_->outputs)
    {
        output_names_.emplace_back(output.name);
    }

    // Get a stream for the graph
    auto graph_stream = loader_->get_istream_for_file("0/data/model.pb");
    if (graph_stream)
    {
        load_frozen_graph(*graph_stream);
    }
    else
    {
        load_saved_model();
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

int64_t TensorflowNeuropodBackend::get_callable(const std::map<std::string, tensorflow::Tensor> &tensor_feeds,
                                                const std::map<std::string, std::string> &       tensor_fetches)
{
    tensorflow::Session::CallableHandle handle{};

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
            // item.first is the node name in the TF graph
            opts.add_feed(item.first);

            // TODO(vip): Once we explicitly control devices, do something like this:
            // opts.mutable_feed_devices()->insert({item, device_name});
        }

        for (const auto &item : tensor_fetches)
        {
            // item.first is the node name in the TF graph
            opts.add_fetch(item.first);
        }

        // Make the callable using the options we set above
        // Note: this callable will be released in the destructor
        check_tf_status(session_->MakeCallable(opts, &handle));

        // Add it to our cache
        callable_handle_cache_[cache_key] = handle;
    }

    return handle;
}

// Run inference with a set of requested outputs
std::unique_ptr<NeuropodValueMap> TensorflowNeuropodBackend::infer_internal(
    const NeuropodValueMap &inputs, const std::vector<std::string> &requested_outputs)
{
    // In TensorFlow, a callable is a way of running a subgraph given a set of inputs and
    // outputs. It's very similar to `session_->Run` except it has support for more fine-grained
    // control over tensor devices. See https://github.com/tensorflow/tensorflow/issues/5902
    // for more details.

    // Fetches and feeds for our callable
    // Note: these are ordered maps to make it easy to cache callables
    // Map from an output node_name to an output_name
    std::map<std::string, std::string> tensor_fetches;

    // Map from an input node_name to a Tensor
    std::map<std::string, tensorflow::Tensor> tensor_feeds;

    // Get the set of outputs we want to compute
    const auto &output_names = !requested_outputs.empty() ? requested_outputs : output_names_;

    // Transform neuropod output names to node names in the graph
    for (const auto &name : output_names)
    {
        const auto node_name = node_name_mapping_.find(name);
        if (node_name == node_name_mapping_.end())
        {
            NEUROPOD_ERROR("Node {} not found in node_name_mapping. "
                           "Ensure that all items in the input/output spec have a corresponding item "
                           "in the node_name_mapping.",
                           name);
        }

        // Add this node name as an output of the subgraph we want to run
        tensor_fetches.emplace(std::make_pair(node_name->second, name));
    }

    // Loop through all the input tensors and setup the inputs
    for (const auto &entry : inputs)
    {
        const auto node_name = node_name_mapping_.find(entry.first);
        if (node_name == node_name_mapping_.end())
        {
            NEUROPOD_ERROR("Node {} not found in node_name_mapping. "
                           "Ensure that all items in the input/output spec have a corresponding item "
                           "in the node_name_mapping.",
                           entry.first);
        }

        // Get the TensorFlow tensor from the Neuropod tensor
        const auto &input_data =
            std::dynamic_pointer_cast<NativeDataContainer<tensorflow::Tensor &>>(entry.second)->get_native_data();

        // Add this node name as an input to the subgraph we want to run
        tensor_feeds.emplace(std::make_pair(node_name->second, input_data));
    }

    // Create a callable handle and a vector to store our outputs
    tensorflow::Session::CallableHandle handle = get_callable(tensor_feeds, tensor_fetches);
    std::vector<tensorflow::Tensor>     outputs;

    // Setup the inputs
    std::vector<tensorflow::Tensor> tf_inputs;
    tf_inputs.reserve(tensor_feeds.size());
    for (auto &item : tensor_feeds)
    {
        tf_inputs.emplace_back(std::move(item.second));
    }

    // Run the callable
    check_tf_status(session_->RunCallable(handle, tf_inputs, &outputs, nullptr));

    // Read the outputs and wrap them in `NeuropodTensor`s
    auto   to_return = stdx::make_unique<NeuropodValueMap>();
    size_t position  = 0;
    for (const auto &item : tensor_fetches)
    {
        const auto &output_name   = item.second;
        auto &      output_tensor = outputs[position++];
        const auto  tensor_type   = get_neuropod_type_from_tf_type(output_tensor.dtype());
        (*to_return)[output_name] = make_tensor<TensorflowNeuropodTensor>(tensor_type, std::move(output_tensor));
    }

    return to_return;
}

REGISTER_NEUROPOD_BACKEND(TensorflowNeuropodBackend, "tensorflow", STR(TENSORFLOW_VERSION))

} // namespace neuropod
