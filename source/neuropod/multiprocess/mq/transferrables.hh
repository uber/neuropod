//
// Uber, Inc. (c) 2020
//

#pragma once

#include <boost/any.hpp>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace neuropod
{

namespace detail
{

// `Transferrables` are items that must be kept in scope while a message is in transit
// Usually, these contain data that is stored in shared memory and is
// used to ensure the sending process maintains a reference to the
// data until the receiving process is done loading it.
// This is used to implement cross-process "moves" of data in shared memory
using Transferrables = std::vector<boost::any>;

// Used to keep track of items in transit and free them when we receive a DONE message
class TransferrableController
{
private:
    // Stores items in transit
    // A mapping from message_id to items that need to be kept in scope
    // until a DONE message is received for that message id
    std::unordered_multimap<uint64_t, boost::any> in_transit_;
    std::mutex                                    in_transit_mutex_;

public:
    TransferrableController();
    ~TransferrableController();

    // Adds transferrables for a message with id `msg_id`
    void add(uint64_t msg_id, Transferrables items);

    // Mark a message as "done". This removes any transferrables that were
    // associated with that message
    void done(uint64_t msg_id);

    // Returns how many messages with transferrable items are still
    // in transit
    size_t size();
};

} // namespace detail

} // namespace neuropod