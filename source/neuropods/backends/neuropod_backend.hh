//
// Uber, Inc. (c) 2018
//

#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "neuropods/internal/backend_registration.hh"
#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/internal/tensor_types.hh"

namespace neuropods
{

class NeuropodTensor;
struct TensorStore;

// The interface that every neuropod backend implements
class NeuropodBackend
{
protected:
    // The neuropod model config (if any)
    std::unique_ptr<ModelConfig> model_config_;

public:
    NeuropodBackend() {}

    // An optional ModelConfig. See `get_neuropod_model_config` below for more details
    NeuropodBackend(std::unique_ptr<ModelConfig> model_config) : model_config_(std::move(model_config)) {}

    virtual ~NeuropodBackend() {}

    // Allocate a tensor of a specific type
    virtual std::unique_ptr<NeuropodTensor> allocate_tensor(const std::string &         node_name,
                                                            const std::vector<int64_t> &input_dims,
                                                            TensorType                  tensor_type)
        = 0;

    // Allocate a tensor of a specific type and wrap existing memory.
    // Note: Some backends may have specific alignment requirements (e.g. tensorflow).
    // To support all the built-in backends, `data` should be aligned to 64 bytes.
    // `deleter` will be called with a pointer to `data` when the tensor is
    // deallocated
    virtual std::unique_ptr<NeuropodTensor> tensor_from_memory(const std::string &         node_name,
                                                               const std::vector<int64_t> &input_dims,
                                                               TensorType                  tensor_type,
                                                               void *                      data,
                                                               const Deleter &             deleter)
        = 0;

    // Run inference
    virtual std::unique_ptr<TensorStore> infer(const std::unordered_set<std::shared_ptr<NeuropodTensor>> &inputs) = 0;

    // The backend retains ownership of the model config
    // This can be a nullptr if there is no associated Neuropod config
    // (e.g. if a TF or TorchScript model was directly loaded instead of using a
    // Neuropod package)
    ModelConfig * get_neuropod_model_config() const { return model_config_.get(); }
};

template<template <class> class TensorImpl>
class NeuropodBackendWithDefaultAllocator : public virtual NeuropodBackend
{
public:
    std::unique_ptr<NeuropodTensor> allocate_tensor(const std::string &         node_name,
                                                    const std::vector<int64_t> &input_dims,
                                                    TensorType                  tensor_type)
    {
        return make_tensor<TensorImpl>(tensor_type, node_name, input_dims);
    }

    std::unique_ptr<NeuropodTensor> tensor_from_memory(const std::string &         node_name,
                                                       const std::vector<int64_t> &input_dims,
                                                       TensorType                  tensor_type,
                                                       void *                      data,
                                                       const Deleter &             deleter)
    {
        return make_tensor_no_string<TensorImpl>(tensor_type, node_name, input_dims, data, deleter);
    }
};

} // namespace neuropods
