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

#include "neuropod/internal/cuda_device_mapping.hh"

#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/logging.hh"

#include <unordered_map>

#include <dlfcn.h>

namespace neuropod
{

namespace
{

using cudaError_t  = int;
using nvmlReturn_t = int;
using nvmlDevice_t = void *;

// __host__​__device__​cudaError_t cudaGetDeviceCount ( int* count )
cudaError_t (*cudaGetDeviceCount)(int *);

// __host__​cudaError_t cudaDeviceGetPCIBusId ( char* pciBusId, int  len, int  device )
cudaError_t (*cudaDeviceGetPCIBusId)(char *, int, int);

// __host__​__device__​cudaError_t cudaGetLastError ( void )
cudaError_t (*cudaGetLastError)();

// nvmlReturn_t nvmlInit ( void )
nvmlReturn_t (*nvmlInit)();

// nvmlReturn_t nvmlDeviceGetHandleByPciBusId ( const char* pciBusId, nvmlDevice_t* device )
nvmlReturn_t (*nvmlDeviceGetHandleByPciBusId)(const char *, nvmlDevice_t *);

// nvmlReturn_t nvmlDeviceGetUUID ( nvmlDevice_t device, char* uuid, unsigned int  length )
nvmlReturn_t (*nvmlDeviceGetUUID)(nvmlDevice_t, char *, unsigned int);

// const DECLDIR char* nvmlErrorString ( nvmlReturn_t result )
const char *(*nvmlErrorString)(nvmlReturn_t);

// Try to load CUDA
bool load_cuda()
{
    void *cuda_handle = nullptr;

    // CUDA suffixes in priority order
    const auto cuda_suffixes = {"", ".10.2", ".10.1", ".10.0", ".9.0"};
    for (const std::string &suffix : cuda_suffixes)
    {
        const auto sopath = "libcudart.so" + suffix;
        SPDLOG_TRACE("Trying to load CUDA runtime: {}", sopath);
        cuda_handle = dlopen(sopath.c_str(), RTLD_LAZY);
        if (cuda_handle)
        {
            break;
        }
    }

    if (!cuda_handle)
    {
        // Couldn't load the CUDA runtime
        SPDLOG_DEBUG("Neuropod could not load the CUDA runtime");
        return false;
    }

    // Load the functions we care about
    cudaGetDeviceCount = reinterpret_cast<cudaError_t (*)(int *)>(dlsym(cuda_handle, "cudaGetDeviceCount"));

    cudaDeviceGetPCIBusId =
        reinterpret_cast<cudaError_t (*)(char *, int, int)>(dlsym(cuda_handle, "cudaDeviceGetPCIBusId"));

    cudaGetLastError = reinterpret_cast<cudaError_t (*)()>(dlsym(cuda_handle, "cudaGetLastError"));

    return true;
}

// Try to load NVML
bool load_nvml()
{
    void *nvml_handle = nullptr;

    // NVML suffixes in priority order
    const auto nvml_suffixes = {"", ".1"};
    for (const std::string &suffix : nvml_suffixes)
    {
        const auto sopath = "libnvidia-ml.so" + suffix;
        SPDLOG_TRACE("Trying to load NVML: {}", sopath);
        nvml_handle = dlopen(sopath.c_str(), RTLD_LAZY);
        if (nvml_handle)
        {
            break;
        }
    }

    if (!nvml_handle)
    {
        // Couldn't load NVML
        SPDLOG_DEBUG("Neuropod could not load NVML");
        return false;
    }

    // Load the functions we care about
    nvmlInit = reinterpret_cast<nvmlReturn_t (*)()>(dlsym(nvml_handle, "nvmlInit"));

    nvmlDeviceGetHandleByPciBusId = reinterpret_cast<nvmlReturn_t (*)(const char *, nvmlDevice_t *)>(
        dlsym(nvml_handle, "nvmlDeviceGetHandleByPciBusId"));

    nvmlDeviceGetUUID =
        reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int)>(dlsym(nvml_handle, "nvmlDeviceGetUUID"));

    nvmlErrorString = reinterpret_cast<const char *(*) (nvmlReturn_t)>(dlsym(nvml_handle, "nvmlErrorString"));

    auto err = nvmlInit();
    if (err != 0 /* NVML_SUCCESS */)
    {
        SPDLOG_ERROR("Error when initializing NVML: {}", nvmlErrorString(err));
        return false;
    }

    return true;
}

// Gets a mapping from a CUDA GPU index to a UUID corresponding to the GPU
// This is a standard id that is not affected by CUDA_VISIBLE_DEVICES so we can
// use it to have stable IDs across processes (e.g. for OPE)
std::unordered_map<int, std::string> get_id_mapping()
{
    // Make sure our logging is initialized
    init_logging();

    if (!load_cuda() || !load_nvml())
    {
        // Couldn't load CUDA or NVML so we can't do anything else
        return {};
    }

    // Get device count
    // Based on https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDAFunctions.h#L19
    int device_count;
    int err = cudaGetDeviceCount(&device_count);

    // Check if CUDA gave us an error
    if (err != 0 /* cudaSuccess */)
    {
        // Clear out the error state, so we don't spuriously trigger someone else.
        cudaGetLastError();
        SPDLOG_DEBUG("Error when getting number of GPU devices");
        return {};
    }

    // Check if we have a GPU
    if (device_count <= 0)
    {
        SPDLOG_DEBUG("No GPUs available");
        return {};
    }

    std::unordered_map<int, std::string> id_mapping;
    for (int i = 0; i < device_count; i++)
    {
        // Get the UUID from the device ID

        // At most 13 chars according to
        // https://docs.nvidia.com/cuda/archive/9.0/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gea264dad3d8c4898e0b82213c0253def
        char pciBusId[13];
        err = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), i);
        if (err != 0 /* cudaSuccess */)
        {
            // Clear out the error state, so we don't spuriously trigger someone else.
            cudaGetLastError();
            SPDLOG_ERROR("Error when getting pciBusId for GPU {}", i);
            return {};
        }

        // Get an NVML device handle
        nvmlDevice_t device;
        err = nvmlDeviceGetHandleByPciBusId(pciBusId, &device);
        if (err != 0 /* NVML_SUCCESS */)
        {
            SPDLOG_ERROR("NVML error when getting device from pciBusId: {}", nvmlErrorString(err));
        }

        // Get a UUID from the handle
        // At most 80 chars according to
        // https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g84dca2d06974131ccec1651428596191
        char uuid[80];
        err = nvmlDeviceGetUUID(device, uuid, sizeof(uuid));
        if (err != 0 /* NVML_SUCCESS */)
        {
            SPDLOG_ERROR("NVML error when getting uuid from device: {}", nvmlErrorString(err));
        }

        SPDLOG_INFO("Found GPU {} with UUID {}", i, uuid);

        id_mapping.emplace(i, uuid);
    }

    return id_mapping;
}

const auto cuda_id_to_uuid = get_id_mapping();

} // namespace

std::string get_gpu_uuid(int cuda_id)
{
    if (cuda_id_to_uuid.empty())
    {
        SPDLOG_DEBUG(
            "No GPUs available, but requested CUDA ID {}. (This message is expected if the machine does not have GPUs)",
            cuda_id);
    }

    auto it = cuda_id_to_uuid.find(cuda_id);
    if (it != cuda_id_to_uuid.end())
    {
        return it->second;
    }

    SPDLOG_WARN("Didn't find a GPU corresponding to the requested CUDA ID {}. Returning empty UUID", cuda_id);
    return "";
}

} // namespace neuropod
