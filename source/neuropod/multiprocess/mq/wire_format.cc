//
// Uber, Inc. (c) 2020
//

#include "neuropod/multiprocess/mq/wire_format.hh"

namespace neuropod
{

namespace detail
{

// The allocator used to allocate SHM for serialization/deserialization into the wire format
SHMAllocator wire_format_shm_allocator;

} // namespace detail

} // namespace neuropod