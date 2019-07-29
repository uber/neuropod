//
// Uber, Inc. (c) 2019
//

#pragma once

#include <functional>

namespace neuropods
{

// Used to deallocate user provided memory
using Deleter = std::function<void(void *)>;

// Run a deleter with a specific handle and then delete the deleter
void run_deleter(void *handle);

// Register a deleter function for some memory. Returns an opaque
// handle
void *register_deleter(const Deleter &deleter, void *data);

} // namespace neuropods
