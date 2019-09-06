//
// Uber, Inc. (c) 2019
//

#include "neuropods/multiprocess/message_utils.hh"
#include "neuropods/multiprocess/multiprocess_worker.hh"

#include "gtest/gtest.h"

TEST(test_multiprocess_worker, invalid_transition)
{
    // TODO(vip): maybe dynamically generate a queue name?
    // Open the control channels
    const std::string control_queue_name = "test_multiprocess_worker_invalid_transition";
    ipc::message_queue to_worker(ipc::open_or_create, ("neuropod_" + control_queue_name + "_tw").c_str(), 20, sizeof(neuropods::control_message));
    ipc::message_queue from_worker(ipc::open_or_create, ("neuropod_" + control_queue_name + "_fw").c_str(), 20, sizeof(neuropods::control_message));

    // Send a message (that is invalid as the first message)
    neuropods::send_message(to_worker, neuropods::INFER);

    // Run the worker loop and expect an error
    EXPECT_ANY_THROW(neuropods::multiprocess_worker_loop(control_queue_name));

    // Cleanup
    ipc::message_queue::remove(("neuropod_" + control_queue_name + "_tw").c_str());
    ipc::message_queue::remove(("neuropod_" + control_queue_name + "_fw").c_str());
}

TEST(test_multiprocess_worker, shutdown)
{
    // TODO(vip): maybe dynamically generate a queue name?
    // Open the control channels
    const std::string control_queue_name = "test_multiprocess_worker_shutdown";
    ipc::message_queue to_worker(ipc::open_or_create, ("neuropod_" + control_queue_name + "_tw").c_str(), 20, sizeof(neuropods::control_message));
    ipc::message_queue from_worker(ipc::open_or_create, ("neuropod_" + control_queue_name + "_fw").c_str(), 20, sizeof(neuropods::control_message));

    // Send a shutdown message
    neuropods::send_message(to_worker, neuropods::SHUTDOWN);

    // The worker loop should run and terminate with no errors
    neuropods::multiprocess_worker_loop(control_queue_name);

    // Cleanup
    ipc::message_queue::remove(("neuropod_" + control_queue_name + "_tw").c_str());
    ipc::message_queue::remove(("neuropod_" + control_queue_name + "_fw").c_str());
}
