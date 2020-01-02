//
// Uber, Inc. (c) 2019
//

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
