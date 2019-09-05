//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/multiprocess/control_messages.hh"

#include <boost/interprocess/ipc/message_queue.hpp>

namespace ipc = boost::interprocess;

namespace neuropods
{

// Utility to send a message with no content to a message queue
void send_message(ipc::message_queue &queue, MessageType type);

// Utility to send a NeuropodValueMap to a message queue
void send_message(ipc::message_queue &queue, MessageType type, const NeuropodValueMap &data);

} // namespace neuropods
