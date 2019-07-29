//
// Uber, Inc. (c) 2018
//

#pragma once

#include <tensorflow/c/c_api.h>

#include <memory>
#include <sstream>
#include <stdexcept>

namespace neuropods
{

/// TF status deleter type
struct tf_status_deleter
{
    void operator()(TF_Status *status) { TF_DeleteStatus(status); }
};

typedef std::unique_ptr<TF_Status, tf_status_deleter> TF_StatusPtr;

/// TF graph deleter type
struct tf_graph_deleter
{
    void operator()(TF_Graph *graph) { TF_DeleteGraph(graph); }
};

typedef std::unique_ptr<TF_Graph, tf_graph_deleter> TF_GraphPtr;

/// TF session deleter type
struct tf_session_deleter
{
    void operator()(TF_Session *session)
    {
        std::unique_ptr<TF_Status, tf_status_deleter> status(TF_NewStatus());
        TF_CloseSession(session, status.get());
        if (TF_GetCode(status.get()) != TF_OK)
        {
            NEUROPOD_ERROR("Failed to close session: " << TF_Message(status.get()));
        }
        TF_DeleteSession(session, status.get());
        if (TF_GetCode(status.get()) != TF_OK)
        {
            NEUROPOD_ERROR("Failed to delete session: " << TF_Message(status.get()));
        }
    }
};

typedef std::unique_ptr<TF_Session, tf_session_deleter> TF_SessionPtr;

/// TF buffer deleter type
struct tf_buffer_deleter
{
    void operator()(TF_Buffer *buffer) { TF_DeleteBuffer(buffer); }
};

typedef std::unique_ptr<TF_Buffer, tf_buffer_deleter> TF_BufferPtr;

/// TF graph options deleter type
struct tf_graph_options_deleter
{
    void operator()(TF_ImportGraphDefOptions *options) { TF_DeleteImportGraphDefOptions(options); }
};

typedef std::unique_ptr<TF_ImportGraphDefOptions, tf_graph_options_deleter> TF_ImportGraphDefOptionsPtr;

/// TF session options deleter type
struct tf_session_options_deleter
{
    void operator()(TF_SessionOptions *options) { TF_DeleteSessionOptions(options); }
};

typedef std::unique_ptr<TF_SessionOptions, tf_session_options_deleter> TF_SessionOptionsPtr;

} // namespace neuropods
