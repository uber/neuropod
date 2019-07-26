//
// Uber, Inc. (c) 2019
//

#pragma once

#include "neuropods/neuropods.hh"

#include <string>

namespace neuropods
{

// Start a worker process and run the neuropod in the worker
// (using shared memory to communicate between the processes)
// See the comment about `free_memory_every_cycle` below.
std::unique_ptr<Neuropod> load_neuropod_in_new_process(const std::string &neuropod_path, bool free_memory_every_cycle = true);

// Run the neuropod in an existing worker
// (using shared memory to communicate between the processes)
// Internally, this uses a shared memory allocator that reuses blocks of memory if possible.
// Therefore memory isn't necessarily allocated during each cycle as blocks may be reused.
//
// If free_memory_every_cycle is set, then unused shared memory will be freed every cycle
// This is useful for simple inference, but for code that is pipelined
// (e.g. generating inputs for cycle t + 1 during the inference of cycle t), this may not
// be desirable.
//
// If free_memory_every_cycle is false, the user is responsible for periodically calling
// neuropods::free_unused_shm_blocks()
std::unique_ptr<Neuropod> load_neuropod_in_worker(const std::string &neuropod_path, const std::string &control_queue_name, bool free_memory_every_cycle = true);

} // namespace neuropods
