/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Mostly from https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/cc/saved_model/loader.cc
// with some modifications

#include "neuropod/backends/tensorflow/saved_model/loader.h"

#include "neuropod/backends/tensorflow/saved_model/constants.h"
#include "neuropod/backends/tensorflow/saved_model/loader_util.h"
#include "neuropod/backends/tensorflow/saved_model/reader.h"
#include "neuropod/backends/tensorflow/tf_utils.hh"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

#include <unordered_set>

namespace tensorflow
{
namespace
{

// Ensure that constant tensors loaded from the saved model have valid shape.
// Also ensure that constant nodes have a value assigned to them.
// TODO(b/154763635): this is temporary and will be replaced with a better audit
static Status ValidateNode(const NodeDef &node)
{
    const auto node_iterator = node.attr().find("value");
    if (node_iterator != node.attr().end())
    {
        AttrValue node_value = node_iterator->second;
        if (node_value.has_tensor())
        {
            const PartialTensorShape node_shape(node_value.tensor().tensor_shape());
            if (node_shape.num_elements() < 0)
            {
                return errors::FailedPrecondition("Saved model contains node \"",
                                                  node.name(),
                                                  "\" (op \"",
                                                  node.op(),
                                                  "\") which initializes from a tensor with ",
                                                  node_shape.num_elements(),
                                                  " elements");
            }
        }
    }
    else if (node.op() == "Const")
    {
        return errors::FailedPrecondition("Saved model contains node \"",
                                          node.name(),
                                          "\" which is a constant tensor but no value has been provided");
    }
    return Status::OK();
}

static Status ValidateSavedTensors(const GraphDef &graph_def)
{
    for (const auto &node : graph_def.node())
    {
        TF_RETURN_IF_ERROR(ValidateNode(node));
    }

    if (graph_def.has_library())
    {
        const FunctionDefLibrary &library = graph_def.library();
        for (const auto &function : library.function())
        {
            for (const auto &node : function.node_def())
            {
                TF_RETURN_IF_ERROR(ValidateNode(node));
            }
        }
    }

    return Status::OK();
}

// NEUROPOD MODIFICATION (to be compatible with TF 1.x and TF 2.x)
#if TF_MAJOR_VERSION > 1
using TFStringType = tensorflow::tstring;
#else
using TFStringType = std::string;
#endif

Tensor CreateStringTensor(const string &value)
{
    Tensor tensor(DT_STRING, TensorShape({}));
    tensor.scalar<TFStringType>()() = value;
    return tensor;
}

void AddAssetsTensorsToInputs(const StringPiece                       export_dir,
                              const std::vector<AssetFileDef> &       asset_file_defs,
                              std::vector<std::pair<string, Tensor>> *inputs)
{
    if (asset_file_defs.empty())
    {
        return;
    }
    for (auto &asset_file_def : asset_file_defs)
    {
        Tensor assets_file_path_tensor =
            CreateStringTensor(io::JoinPath(export_dir, kSavedModelAssetsDirectory, asset_file_def.filename()));
        inputs->push_back({asset_file_def.tensor_info().name(), assets_file_path_tensor});
    }
}

// Like Session::Run(), but uses the Make/Run/ReleaseCallable() API to avoid
// leaving behind non-GC'ed state.
//
// Detailed motivation behind this approach, from ashankar@:
//
// Each call to Session::Run() that identifies a new subgraph (based on feeds
// and fetches) creates some datastructures that live as long as the session
// (the partitioned graph, associated executors etc.).
//
// A pathological case of this would be if say the initialization op
// (main_op/legacy_init_op) involves the use of a large constant. Then we
// allocate memory for that large constant that will just stick around till the
// session dies. With this Callable mechanism, that memory will be released
// right after ReleaseCallable returns.
//
// However, the resource manager state remains.
Status RunOnce(const RunOptions &                            run_options,
               const std::vector<std::pair<string, Tensor>> &inputs,
               const std::vector<string> &                   output_tensor_names,
               const std::vector<string> &                   target_node_names,
               std::vector<Tensor> *                         outputs,
               RunMetadata *                                 run_metadata,
               Session *                                     session)
{
    CallableOptions     callable_options;
    std::vector<Tensor> feed_tensors;
    *callable_options.mutable_run_options() = run_options;
    for (const auto &input : inputs)
    {
        const string &name   = input.first;
        const Tensor &tensor = input.second;
        callable_options.add_feed(name);
        feed_tensors.push_back(tensor);
    }
    for (const string &output_tensor_name : output_tensor_names)
    {
        callable_options.add_fetch(output_tensor_name);
    }
    for (const string &target_node_name : target_node_names)
    {
        callable_options.add_target(target_node_name);
    }

    Session::CallableHandle callable_handle;
    TF_RETURN_IF_ERROR(session->MakeCallable(callable_options, &callable_handle));
    const Status run_status = session->RunCallable(callable_handle, feed_tensors, outputs, run_metadata);
    // Be sure to call ReleaseCallable() regardless of the outcome of
    // RunCallable().
    session->ReleaseCallable(callable_handle).IgnoreError();
    return run_status;
}

// RunInitOp will return OK if the initialization op was run successfully.
// An empty init_op_name indicates that there are no init ops to run.
Status RunInitOp(const RunOptions &               run_options,
                 const string &                   export_dir,
                 const std::vector<AssetFileDef> &asset_file_defs,
                 Session *                        session,
                 const string &                   init_op_name)
{
    if (!init_op_name.empty())
    {
        LOG(INFO) << "Running initialization op on SavedModel bundle at path: " << export_dir;
        std::vector<std::pair<string, Tensor>> inputs;
        AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);
        RunMetadata run_metadata;
        return RunOnce(run_options, inputs, {}, {init_op_name}, nullptr /* outputs */, &run_metadata, session);
    }
    return Status::OK();
}

// From https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/core/util/tensor_bundle/naming.cc
string MetaFilename(StringPiece prefix)
{
    return strings::Printf("%.*s.index", static_cast<int>(prefix.size()), prefix.data());
}

Status RunRestore(const RunOptions &               run_options,
                  const string &                   export_dir,
                  const StringPiece                restore_op_name,
                  const StringPiece                variable_filename_const_op_name,
                  const std::vector<AssetFileDef> &asset_file_defs,
                  Session *                        session)
{
    LOG(INFO) << "Restoring SavedModel bundle.";
    // Find path to variables to be restored in export directory.
    const string variables_directory = io::JoinPath(export_dir, kSavedModelVariablesDirectory);
    // Check for saver checkpoints in v2 format. Models exported in the checkpoint
    // v2 format will have a variables.index file. The corresponding
    // variables are stored in the variables.data-?????-of-????? files.
    const string variables_index_path = io::JoinPath(variables_directory, MetaFilename(kSavedModelVariablesFilename));
    if (!Env::Default()->FileExists(variables_index_path).ok())
    {
        LOG(INFO) << "The specified SavedModel has no variables; no checkpoints "
                     "were restored. File does not exist: "
                  << variables_index_path;
        return Status::OK();
    }
    const string variables_path = io::JoinPath(variables_directory, kSavedModelVariablesFilename);

    // Add variables to the graph.
    Tensor variables_path_tensor(DT_STRING, TensorShape({}));
    variables_path_tensor.scalar<TFStringType>()() = variables_path;

    std::vector<std::pair<string, Tensor>> inputs = {{string(variable_filename_const_op_name), variables_path_tensor}};

    AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);

    RunMetadata run_metadata;
    return RunOnce(run_options, inputs, {}, {string(restore_op_name)}, nullptr /* outputs */, &run_metadata, session);
}

Status LoadSavedModelInternal(const SessionOptions &            session_options,
                              const RunOptions &                run_options,
                              const string &                    export_dir,
                              const std::unordered_set<string> &tags,
                              SavedModelBundle *const           bundle)
{
    TF_RETURN_IF_ERROR(ReadMetaGraphDefFromSavedModel(export_dir, tags, &bundle->meta_graph_def));

    // NEUROPOD MODIFICATION
    // DebugInfo isn't available in all supported TF versions and is not critical to running a
    // SavedModel so we ignore it for now
    //
    //   TF_RETURN_IF_ERROR(
    //       ReadSavedModelDebugInfoIfPresent(export_dir, &bundle->debug_info));
    TF_RETURN_IF_ERROR(LoadMetagraphIntoSession(session_options, bundle->meta_graph_def, &bundle->session));
    TF_RETURN_IF_ERROR(RestoreSession(run_options, bundle->meta_graph_def, export_dir, &bundle->session));
    return Status::OK();
}

} // namespace

SavedModelBundleInterface::~SavedModelBundleInterface() {}

Status LoadMetagraphIntoSession(const SessionOptions &    session_options,
                                MetaGraphDef &            meta_graph,
                                std::unique_ptr<Session> *session)
{
    Session *session_p = nullptr;
    TF_RETURN_IF_ERROR(NewSession(session_options, &session_p));
    session->reset(session_p);

    GraphDef &graph = *meta_graph.mutable_graph_def();
    TF_RETURN_IF_ERROR(ValidateSavedTensors(graph));

    // NEUROPOD MODIFICATION
    // This moves the graph to the appropriate device before loading it
    neuropod::move_graph_to_device(graph, *(session->get()), 0);

    return (*session)->Create(graph);
}

Status LoadSavedModel(const SessionOptions &            session_options,
                      const RunOptions &                run_options,
                      const string &                    export_dir,
                      const std::unordered_set<string> &tags,
                      SavedModelBundle *const           bundle)
{
    return LoadSavedModelInternal(session_options, run_options, export_dir, tags, bundle);
}

Status RestoreSession(const RunOptions &        run_options,
                      const MetaGraphDef &      meta_graph,
                      const string &            export_dir,
                      std::unique_ptr<Session> *session)
{
    std::vector<AssetFileDef> asset_file_defs;
    TF_RETURN_IF_ERROR(internal::GetAssetFileDefs(meta_graph, &asset_file_defs));
    if (meta_graph.has_saver_def())
    {
        TF_RETURN_IF_ERROR(RunRestore(run_options,
                                      export_dir,
                                      meta_graph.saver_def().restore_op_name(),
                                      meta_graph.saver_def().filename_tensor_name(),
                                      asset_file_defs,
                                      session->get()));
    }

    string init_op_name;
    TF_RETURN_IF_ERROR(internal::GetInitOp(export_dir, meta_graph, &init_op_name));
    TF_RETURN_IF_ERROR(RunInitOp(run_options, export_dir, asset_file_defs, session->get(), init_op_name));

    return Status::OK();
}

bool MaybeSavedModelDirectory(const string &export_dir)
{
    const string saved_model_pb_path    = io::JoinPath(export_dir, kSavedModelFilenamePb);
    const string saved_model_pbtxt_path = io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
    return Env::Default()->FileExists(saved_model_pb_path).ok() ||
           Env::Default()->FileExists(saved_model_pbtxt_path).ok();
}

} // namespace tensorflow
