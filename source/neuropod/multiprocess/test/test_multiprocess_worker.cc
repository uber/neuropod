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

#include "gtest/gtest.h"
#include "neuropod/multiprocess/ipc_control_channel.hh"
#include "neuropod/multiprocess/multiprocess_worker.hh"

TEST(test_multiprocess_worker, shutdown)
{
    // TODO(vip): maybe dynamically generate a queue name?
    // Open the control channels
    const std::string           control_queue_name = "test_multiprocess_worker_shutdown";
    neuropod::IPCControlChannel control_channel(control_queue_name, neuropod::MAIN_PROCESS);

    // Send a shutdown message
    control_channel.send_message(neuropod::SHUTDOWN);

    // The worker loop should run and terminate with no errors
    neuropod::multiprocess_worker_loop(control_queue_name);

    // Cleanup
    control_channel.cleanup();
}
