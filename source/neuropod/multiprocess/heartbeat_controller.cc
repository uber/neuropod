//
// Uber, Inc. (c) 2020
//

#include "neuropod/multiprocess/heartbeat_controller.hh"

#include "neuropod/multiprocess/ipc_control_channel.hh"

namespace neuropod
{

// Starts a new thread to send a heartbeat
HeartbeatController::HeartbeatController(IPCControlChannel &control_channel, size_t interval_ms)
    : send_heartbeat_(true), heartbeat_thread_([this, &control_channel, interval_ms]() {
          // Send a heartbeat every 2 seconds
          while (send_heartbeat_)
          {
              // Note: we're using `try_send_message` instead of `send_message` because
              // we don't want to block if the message queue is full.
              // This is important because it could prevent shutdown and cause the worker
              // and main process to hang.
              //
              // For example:
              // - The main process loads a neuropod using the multiprocess backend (but does not run inference)
              // - The (worker -> main process) message queue fills up with HEARTBEAT messages
              //   (because messages are only read from that queue during inference)
              //
              // - This thread blocks on sending the next HEARTBEAT message
              // - The main process sends a SHUTDOWN message and waits for the worker to shut down
              // - The destructor of `HeartbeatController` runs and runs `join` on this thread
              //
              // This will cause both processes to hang forever because the worker is still blocked on the message
              // queue and the main process is waiting for this process to shutdown.

              // try_send_message is threadsafe so we don't need synchronization here
              control_message msg;
              msg.type = HEARTBEAT;
              if (!control_channel.try_send_message(msg))
              {
                  SPDLOG_DEBUG("OPE: Message queue full - skipped sending heartbeat");
              }

              // This lets us break out of waiting if we're told to shutdown
              std::unique_lock<std::mutex> lk(mutex_);
              cv_.wait_for(lk, std::chrono::milliseconds(interval_ms), [&] { return send_heartbeat_ != true; });
          }
      })
{
}

HeartbeatController::~HeartbeatController()
{
    // Join the heartbeat thread
    {
        std::lock_guard<std::mutex> lk(mutex_);
        send_heartbeat_ = false;
    }

    cv_.notify_all();
    heartbeat_thread_.join();
}

} // namespace neuropod