//
// Uber, Inc. (c) 2020
//

#pragma once

#include <string>

namespace neuropod
{

typedef int NeuropodDevice;
namespace Device
{
constexpr int CPU  = -1;
constexpr int GPU0 = 0;
constexpr int GPU1 = 1;
constexpr int GPU2 = 2;
constexpr int GPU3 = 3;
constexpr int GPU4 = 4;
constexpr int GPU5 = 5;
constexpr int GPU6 = 6;
constexpr int GPU7 = 7;
} // namespace Device

// Runtime options that are available when loading a Neuropod
struct RuntimeOptions
{
    // Whether or not to use out-of-process execution
    // (using shared memory to communicate between the processes)
    bool use_ope = false;

    // These options are only used if use_ope is set to true
    struct OPEOptions
    {
        // Internally, OPE uses a shared memory allocator that reuses blocks of memory if possible.
        // Therefore memory isn't necessarily allocated during each inference cycle as blocks may
        // be reused.
        //
        // If free_memory_every_cycle is set, then unused shared memory will be freed every cycle
        // This is useful for simple inference, but for code that is pipelined
        // (e.g. generating inputs for cycle t + 1 during the inference of cycle t), this may not
        // be desirable.
        //
        // If free_memory_every_cycle is false, the user is responsible for periodically calling
        // neuropod::free_unused_shm_blocks()
        bool free_memory_every_cycle = true;

        // This option can be used to run the neuropod in an existing worker process
        // If this string is empty, a new worker will be started.
        std::string control_queue_name;
    } ope_options;

    // The device to run this Neuropod on.
    // Some devices are defined in the namespace above. For machines with more
    // than 8 GPUs, passing in an index will also work (e.g. `9` for `GPU9`).
    //
    // To attempt to run the model on CPU, set this to `Device::CPU`
    NeuropodDevice visible_device = Device::GPU0;

    // Sometimes, it's important to be able to instantiate a Neuropod without
    // immediately loading the model. If this is set to `false`, the model will
    // not be loaded until the `load_model` method is called on the Neuropod.
    bool load_model_at_construction = true;
};

} // namespace neuropod
