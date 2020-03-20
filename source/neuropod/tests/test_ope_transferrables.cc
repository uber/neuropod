//
// Uber, In (c) 2020
//

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "neuropod/multiprocess/mq/transferrables.hh"

namespace
{

int item_counter = 0;

// A class that increments a counter on construction and decrements it on deletion
// Note: this doesn't need to be threadsafe
struct Item
{
    Item() { item_counter++; }
    Item(const Item &) { item_counter++; }
    Item(Item &&) { item_counter++; }

    ~Item() { item_counter--; }
};

} // namespace

TEST(test_ope_transferrables, basic)
{
    // Create a controller
    neuropod::detail::TransferrableController controller;
    EXPECT_EQ(0, controller.size());

    // Add one item
    controller.add(42, {Item()});
    EXPECT_EQ(1, item_counter);
    EXPECT_EQ(1, controller.size());

    // Add two items
    controller.add(23, {Item(), Item()});
    EXPECT_EQ(3, item_counter);
    EXPECT_EQ(3, controller.size());

    // Clear a non-existing msg_id (which should have no impact)
    controller.done(25);
    EXPECT_EQ(3, item_counter);
    EXPECT_EQ(3, controller.size());

    // Clear the first msg_id (and we should have two items left)
    controller.done(42);
    EXPECT_EQ(2, item_counter);
    EXPECT_EQ(2, controller.size());

    // Clear the second msg_id (and we should have freed all our items)
    controller.done(23);
    EXPECT_EQ(0, item_counter);
    EXPECT_EQ(0, controller.size());
}
