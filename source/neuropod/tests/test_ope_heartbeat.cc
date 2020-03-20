//
// Uber, In (c) 2020
//

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "neuropod/multiprocess/mq/heartbeat.hh"

namespace
{

// A simple "message queue" that just waits for two heartbeat messages
// to be sent before it can shutdown
class TestChannel
{
private:
    std::mutex              mutex_;
    std::condition_variable cv_;

    size_t num_heartbeats_;

public:
    struct WireFormat
    {
        neuropod::detail::QueueMessageType type;
    };

    TestChannel() = default;

    ~TestChannel()
    {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait(lk, [&]() { return num_heartbeats_ >= 2; });
    };

    void send_message(WireFormat msg)
    {
        if (msg.type == neuropod::detail::HEARTBEAT)
        {
            std::lock_guard<std::mutex> lk(mutex_);
            num_heartbeats_++;
            cv_.notify_all();
        }
    }
};

} // namespace

TEST(test_ope_heartbeat, basic)
{
    auto test_channel = neuropod::stdx::make_unique<TestChannel>();

    // Create a controller
    neuropod::detail::HeartbeatController controller(*test_channel);

    // Try to destroy the channel (which will only happen once two heartbeat messages have been sent)
    test_channel.reset();
}
