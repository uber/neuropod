//
// Uber, Inc. (c) 2020
//

#include "neuropod/multiprocess/ipc_control_channel.hh"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace neuropod
{

// Starts a new thread to periodically send a heartbeat
class HeartbeatController
{
private:
    std::atomic_bool        send_heartbeat_;
    std::condition_variable cv_;
    std::mutex              mutex_;
    std::thread             heartbeat_thread_;

public:
    // Start a heartbeat thread
    HeartbeatController(IPCControlChannel &control_channel, size_t interval_ms);

    // Shutdown the heartbeat thread
    ~HeartbeatController();
};

} // namespace neuropod