//
// Uber, Inc. (c) 2020
//

#include "neuropod/multiprocess/mq/transferrables.hh"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace neuropod
{

namespace detail
{

TransferrableController::TransferrableController()   = default;
TransferrableController ::~TransferrableController() = default;

void TransferrableController::add(uint64_t msg_id, Transferrables items)
{
    // Check if there are any transferrable items attached
    if (!items.empty())
    {
        // Insert the transferrables into our map of items to store
        std::lock_guard<std::mutex> lock(in_transit_mutex_);
        for (auto &transferrable : items)
        {
            in_transit_.emplace(msg_id, std::move(transferrable));
        }
    }
}

void TransferrableController::done(uint64_t msg_id)
{
    std::lock_guard<std::mutex> lock(in_transit_mutex_);
    in_transit_.erase(msg_id);
}

size_t TransferrableController::size()
{
    std::lock_guard<std::mutex> lock(in_transit_mutex_);
    return in_transit_.size();
}

} // namespace detail

} // namespace neuropod