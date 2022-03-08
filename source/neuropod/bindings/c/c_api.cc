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

// Inspired by the TensorFlow C API

#include "neuropod/bindings/c/c_api.h"

#include "neuropod/bindings/c/c_api_internal.h"
#include "neuropod/bindings/c/np_status_internal.h"
#include "neuropod/bindings/c/np_tensor_allocator_internal.h"
#include "neuropod/bindings/c/np_valuemap_internal.h"
#include "neuropod/core/generic_tensor.hh"

#include <exception>
#include <string>
#include <vector>

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
void NP_LoadNeuropodWithOpts(const char *             neuropod_path,
                             const NP_RuntimeOptions *options,
                             NP_Neuropod **           model,
                             NP_Status *              status)
{
    try
    {
        *model          = new NP_Neuropod();
        (*model)->model = std::make_unique<neuropod::Neuropod>(neuropod_path);
        NP_ClearStatus(status);
    }
    catch (std::exception &e)
    {
        status->code    = NEUROPOD_ERROR;
        status->message = e.what();
    }
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
NP_RuntimeOptions NP_DefaultRuntimeOptions()
{
    // Use C++ runtime options object to set default values.
    neuropod::RuntimeOptions default_options;
    NP_RuntimeOptions        options;

    options.use_ope                         = default_options.use_ope;
    options.visible_device                  = static_cast<NP_RuntimeOptions::NP_Device>(default_options.visible_device);
    options.load_model_at_construction      = default_options.load_model_at_construction;
    options.disable_shape_and_type_checking = default_options.disable_shape_and_type_checking;

    auto ope_options                     = &options.ope_options;
    ope_options->free_memory_every_cycle = default_options.ope_options.free_memory_every_cycle;
    // Control queue name is empty, a new worker will be started.
    // This is what is expected by default.
    ope_options->control_queue_name[0] = '\0';

    return options;
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
void NP_LoadNeuropod(const char *neuropod_path, NP_Neuropod **model, NP_Status *status)
{
    const auto &options = NP_DefaultRuntimeOptions();
    NP_LoadNeuropodWithOpts(neuropod_path, &options, model, status);
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
void NP_FreeNeuropod(NP_Neuropod *model)
{
    delete model;
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
void NP_Infer(NP_Neuropod *model, const NP_NeuropodValueMap *inputs, NP_NeuropodValueMap **outputs, NP_Status *status)
{
    NP_InferWithRequestedOutputs(model, inputs, 0, nullptr, outputs, status);
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
void NP_InferWithRequestedOutputs(NP_Neuropod *              model,
                                  const NP_NeuropodValueMap *inputs,
                                  size_t                     noutputs,
                                  const char **              requested_outputs,
                                  NP_NeuropodValueMap **     outputs,
                                  NP_Status *                status)
{
    try
    {
        // By default empty collection of requested_ouputs expected.
        // Note that it copies C-strings (0-terminated) from given array.
        // User must guarantee that noutputs and requested_outputs address valid data..
        std::vector<std::string> rout;
        if (requested_outputs != nullptr)
        {
            for (size_t i = 0; i < noutputs; ++i)
            {
                rout.emplace_back(requested_outputs[i]);
            }
        }
        *outputs         = new NP_NeuropodValueMap();
        (*outputs)->data = std::move(*model->model->infer(inputs->data, rout));
        NP_ClearStatus(status);
    }
    catch (std::exception &e)
    {
        status->code    = NEUROPOD_ERROR;
        status->message = e.what();
    }
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
const char *NP_GetName(NP_Neuropod *model)
{
    return model->model->get_name().c_str();
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
const char *NP_GetPlatform(NP_Neuropod *model)
{
    return model->model->get_platform().c_str();
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
size_t NP_GetNumInputs(NP_Neuropod *model)
{
    return model->model->get_inputs().size();
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
size_t NP_GetNumOutputs(NP_Neuropod *model)
{
    return model->model->get_outputs().size();
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
NP_TensorAllocator *NP_GetAllocator(NP_Neuropod *model)
{
    auto out       = new NP_TensorAllocator();
    out->allocator = model->model->get_tensor_allocator();
    return out;
}

// NOLINTNEXTLINE(readability-identifier-naming): Ignore function case for C API methods
NP_TensorAllocator *NP_GetGenericAllocator()
{
    auto out       = new NP_TensorAllocator();
    out->allocator = neuropod::get_generic_tensor_allocator();
    return out;
}
