//
// Uber, In (c) 2020
//

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "neuropod/multiprocess/mq/wire_format.hh"

namespace
{

// An enum for user payload types
// This is not being used as part of the test so we don't specify any payload types
enum MessageType
{
};

} // namespace

TEST(test_ope_wire_format, small_payload)
{
    // Something that fits inline within the payload
    std::vector<std::string> expected = {"some", "vector", "of", "strings"};

    neuropod::detail::WireFormat<MessageType> msg;
    neuropod::detail::Transferrables          transferrables;
    neuropod::detail::serialize_payload(expected, msg, transferrables);

    // This payload should fit within the message so transferrables should
    // be empty
    EXPECT_TRUE(transferrables.empty());

    // Get the payload
    std::vector<std::string> actual;
    neuropod::detail::deserialize_payload(msg, actual);

    EXPECT_EQ(expected, actual);
}

TEST(test_ope_wire_format, large_payload)
{
    // Something that's too large to fit inline within the payload
    std::string large_string(10000, ' ');

    std::vector<std::string> expected = {large_string};

    neuropod::detail::WireFormat<MessageType> msg;
    neuropod::detail::Transferrables          transferrables;
    neuropod::detail::serialize_payload(expected, msg, transferrables);

    // This payload should not fit within the message so transferrables should
    // contain exactly one item
    EXPECT_EQ(1, transferrables.size());

    // Get the payload
    std::vector<std::string> actual;
    neuropod::detail::deserialize_payload(msg, actual);

    EXPECT_EQ(expected, actual);
}
