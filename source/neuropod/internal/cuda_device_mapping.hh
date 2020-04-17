//
// Uber, Inc. (c) 2020
//

#pragma once

#include <string>

namespace neuropod
{

// This returns a UUID string from a cuda ID that's valid in the current process
// Note: The returned string includes the "GPU-" prefix returned by NVML
// Returns an empty string if a GPU corresponding to the requested ID is not available
std::string get_gpu_uuid(int cuda_id);

} // namespace neuropod
