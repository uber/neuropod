/* Copyright (c) 2020 UATC, LLC

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
