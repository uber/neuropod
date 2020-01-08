//
// Uber, Inc. (c) 2019
//

#include "neuropod/internal/deleter.hh"

namespace neuropod
{
namespace
{

struct deleter_wrapper
{
    // A user provided deleter function to run
    Deleter deleter;

    // The data pointer to pass to the deleter
    void *data;
};

} // namespace

void run_deleter(void *handle)
{
    if (handle == nullptr)
    {
        return;
    }

    // Run the deleter and then delete the wrapper
    auto wrapper = static_cast<deleter_wrapper *>(handle);
    wrapper->deleter(wrapper->data);
    delete wrapper;
}

void *register_deleter(const Deleter &deleter, void *data)
{
    // Create a new wrapper and return a handle
    // This is deleted in `run_deleter`
    return new deleter_wrapper({deleter, data});
}

} // namespace neuropod
