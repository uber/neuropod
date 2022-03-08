/* Copyright (c) 2021 The Neuropod Authors

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

#include "neuropod/backends/tensorflow/tf_utils.hh"

#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/logging.hh"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

namespace neuropod
{

// Utility to move a TF graph to a specific device
void move_graph_to_device(tensorflow::GraphDef &graph, tensorflow::Session &session, const NeuropodDevice target)
{
    // Figure out the correct target device
    std::string target_device = "/device:CPU:0";
    if (target != Device::CPU)
    {
        // Get all the available devices
        std::vector<tensorflow::DeviceAttributes> devices;
        check_tf_status(session.ListDevices(&devices));

        // Check if we have any GPUs
        bool found_gpu = std::any_of(devices.begin(), devices.end(), [](const tensorflow::DeviceAttributes &device) {
            return device.device_type() == "GPU";
        });

        // If we have a GPU, update the target device
        if (found_gpu)
        {
            target_device = std::string("/device:GPU:") + std::to_string(target);
        }
    }

    // Iterate through all the nodes in the graph and move them to the target device
    for (auto &node : *graph.mutable_node())
    {
        const auto &node_device = node.device();

        // If a node is on CPU, leave it there
        if (node_device != "/device:CPU:0" && node_device != target_device)
        {
            SPDLOG_TRACE("TF: Moving node {} from device {} to device {}", node.name(), node_device, target_device);
            node.set_device(target_device);
        }
        else
        {
            SPDLOG_TRACE("TF: Leaving node {} on device {}", node.name(), node_device);
        }
    }
}

// Throws an error if `status` is not ok
void check_tf_status(const tensorflow::Status &status)
{
    if (!status.ok())
    {
        NEUROPOD_ERROR("TensorFlow error: {}", status.error_message());
    }
}

} // namespace neuropod
