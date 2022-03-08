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

#include "neuropod/multiprocess/mq/transferrables.hh"

#include "neuropod/internal/logging.hh"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace neuropod::detail
{

TransferrableController::TransferrableController() = default;
TransferrableController::~TransferrableController()
{
    if (!in_transit_.empty())
    {
        SPDLOG_WARN("OPE: Transferrables not empty at shutdown");
    }
}

void TransferrableController::add(uint64_t msg_id, Transferrables items)
{
    // Check if there are any transferrable items attached
    if (!items.empty())
    {
        SPDLOG_TRACE("OPE: Adding {} transferrables for msg with id {}", items.size(), msg_id);
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
    SPDLOG_TRACE("OPE: Clearing transferrables for msg with id {}", msg_id);
    std::lock_guard<std::mutex> lock(in_transit_mutex_);
    in_transit_.erase(msg_id);
}

size_t TransferrableController::size()
{
    std::lock_guard<std::mutex> lock(in_transit_mutex_);
    return in_transit_.size();
}

} // namespace neuropod::detail
