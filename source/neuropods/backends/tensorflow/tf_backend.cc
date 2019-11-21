//
// Uber, Inc. (c) 2018
//

#include "tf_backend.hh"

#include "neuropods/backends/tensorflow/type_utils.hh"

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

// Get a graph node given a node name of the format `name:index`. Namespaces are supported as well.
// If the index is 0, ":0" optional.
// Ex: `some_namespace/input:0`
TF_Output get_graph_node_from_name(const std::string &node_name_with_index, const TF_GraphPtr &graph_)
{
    // `node_name` ends with `:index` where index is an integer
    const auto colon_pos = node_name_with_index.find(":");
    const auto node_name = node_name_with_index.substr(0, colon_pos);

    // The index of the output for the specified op
    int node_index = 0;
    if (colon_pos != std::string::npos)
    {
        node_index = std::stoi(node_name_with_index.substr(colon_pos + 1));
    }

    TF_Operation *oper = TF_GraphOperationByName(graph_.get(), node_name.c_str());
    if (!oper)
    {
        NEUROPOD_ERROR("Operation '" << node_name_with_index << "' is not found");
    }
    // TensorFlow uses TF_Output everywhere, including input placeholders
    // Operations can have several outputs, they are indexed started from 0
    return TF_Output{oper, node_index};
}

} // namespace

TensorflowNeuropodBackend::TensorflowNeuropodBackend(const std::string &           neuropod_path,
                                                     std::unique_ptr<ModelConfig> &model_config,
                                                     const RuntimeOptions &        options)
    : NeuropodBackendWithDefaultAllocator<TensorflowNeuropodTensor>(neuropod_path)
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
    status_.reset(TF_NewStatus());
    for (const auto &item : model_config->custom_ops)
    {
        TF_DeleteLibraryHandle(TF_LoadLibrary(loader_->get_file_path("0/ops/" + item).c_str(), status_.get()));
        check_status();
    }

    // Get the graph and load it
    auto graph_stream = loader_->get_istream_for_file("0/data/model.pb");
    if (!load_graph(*graph_stream))
    {
        NEUROPOD_ERROR("Failed to load TensorFlow graph for neuropod " << neuropod_path);
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

    if (init_ops.size() > 0)
    {
        run_target_ops(init_ops);
    }
}

TensorflowNeuropodBackend::~TensorflowNeuropodBackend() = default;

// Check the status of the last TF API call and throw an exception if
// there was an error
void TensorflowNeuropodBackend::check_status() const
{
    if (TF_GetCode(status_.get()) != TF_OK)
    {
        NEUROPOD_ERROR("Tensorflow error: " << TF_Message(status_.get()));
    }
}

// Load a TensorFlow graph
bool TensorflowNeuropodBackend::load_graph(std::istream &ifs)
{
    if (ifs.good())
    {
        ifs.seekg(0, ifs.end);
        std::streamsize   length = ifs.tellg();
        std::vector<char> buffer(length);

        ifs.seekg(0, ifs.beg);
        ifs.read(buffer.data(), length);
        if (ifs.fail())
        {
            return false;
        }

        status_.reset(TF_NewStatus());
        graph_.reset(TF_NewGraph());
        TF_BufferPtr                graph_def(TF_NewBufferFromString(buffer.data(), length));
        TF_ImportGraphDefOptionsPtr graph_opts(TF_NewImportGraphDefOptions());
        TF_ImportGraphDefOptionsSetPrefix(graph_opts.get(), "");
        TF_GraphImportGraphDef(graph_.get(), graph_def.get(), graph_opts.get(), status_.get());
        check_status();

        session_opts_.reset(TF_NewSessionOptions());

        // TODO(vip): Don't reserve the whole GPU (set allow_growth to True)

        session_.reset(TF_NewSession(graph_.get(), session_opts_.get(), status_.get()));
        check_status();
    }
    else
    {
        return false;
    }

    return true;
}

// Run target ops
void TensorflowNeuropodBackend::run_target_ops(const std::vector<std::string> &target_op_names)
{
    // Loop through all the specified names and setup the outputs
    std::vector<const TF_Operation *> target_ops;
    for (const auto &node_name : target_op_names)
    {
        TF_Operation *oper = TF_GraphOperationByName(graph_.get(), node_name.c_str());
        if (!oper)
        {
            NEUROPOD_ERROR("Operation '" << node_name << "' is not found");
        }

        target_ops.emplace_back(oper);
    }

    // Run the operation
    TF_SessionRun(session_.get(),
                  nullptr, // ignore run options
                  // Input tensors
                  nullptr, // list of handlers on input tensors
                  nullptr, // list of pointers on input tensors
                  0,       // number of inputs
                  // Output tensors
                  nullptr, // list of handlers on output tensors
                  nullptr, // list of pointers on output tensors
                  0,       // number of outputs
                  // Default target operations and run meta-data
                  &target_ops[0],    // target operations are outputs
                  target_ops.size(), // no target operations are provided
                  nullptr,           // ignore run meta-data
                  status_.get());
    check_status();
}

// Run inference
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
        output_node_names.emplace_back(node_name_mapping_[name]);
    }

    // List of TF input operations
    // TF use TF_Output for both inputs and outputs
    std::vector<TF_Output> input_ops;
    // List of pointers to TF input tensors
    std::vector<TF_Tensor *> input_values;

    // Loop through all the input tensors and setup the inputs
    for (const auto &entry : inputs)
    {
        const auto  input_name = node_name_mapping_.at(entry.first);
        const auto &input_data =
            std::dynamic_pointer_cast<NativeDataContainer<TF_Tensor *>>(entry.second)->get_native_data();

        input_ops.emplace_back(get_graph_node_from_name(input_name, graph_));
        input_values.emplace_back(input_data);
    }

    // List of TF output operations
    std::vector<TF_Output> outputs;
    // List of pointers to TF output tensors
    std::vector<TF_Tensor *> output_values;

    // Loop through all the specified names and setup the outputs
    for (const auto &node_name : output_node_names)
    {
        outputs.emplace_back(get_graph_node_from_name(node_name, graph_));
    }

    // Resize the output vector
    output_values.resize(outputs.size(), nullptr);

    // Run inference
    TF_SessionRun(session_.get(),
                  nullptr, // ignore run options
                  // Input tensors
                  &input_ops[0],    // list of handlers on input tensors
                  &input_values[0], // list of pointers on input tensors
                  input_ops.size(), // number of inputs
                  // Output tensors
                  &outputs[0],       // list of handlers on output tensors
                  &output_values[0], // list of pointers on output tensors
                  outputs.size(),    // number of outputs
                  // Default target operations and run meta-data
                  nullptr, // target operations are outputs
                  0,       // no target operations are provided
                  nullptr, // ignore run meta-data
                  status_.get());
    check_status();

    // Read the outputs
    auto to_return = stdx::make_unique<NeuropodValueMap>();
    for (size_t i = 0; i < output_names.size(); ++i)
    {
        const auto &output_name   = output_names[i];
        const auto &output_tensor = output_values[i];
        const auto  tensor_type   = get_neuropod_type_from_tf_type(TF_TensorType(output_tensor));
        (*to_return)[output_name] = make_tensor<TensorflowNeuropodTensor>(tensor_type, output_tensor);
    }

    return to_return;
}

REGISTER_NEUROPOD_BACKEND(TensorflowNeuropodBackend, "tensorflow")

} // namespace neuropods
