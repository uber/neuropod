//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropods/backends/tensor_allocator.hh"
#include "neuropods/internal/backend_registration.hh"
#include "neuropods/internal/deleter.hh"
#include "neuropods/internal/neuropod_tensor.hh"
#include "neuropods/internal/tensor_types.hh"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace neuropods
{

// A map from a tensor name to a pointer to a NeuropodValue
// This is the input and output type of `infer`
using NeuropodValueMap = std::unordered_map<std::string, std::shared_ptr<NeuropodValue>>;

// The interface that every neuropod backend implements
class NeuropodBackend
{
public:
    virtual ~NeuropodBackend() {}

    // Returns an allocator that can allocate tensors compatible with this backend
    virtual std::shared_ptr<NeuropodTensorAllocator> get_tensor_allocator() = 0;

    // Run inference
    virtual std::unique_ptr<NeuropodValueMap> infer(const NeuropodValueMap &inputs) = 0;
};

template <template <class> class TensorImpl>
class NeuropodBackendWithDefaultAllocator : public NeuropodBackend
{
private:
    std::shared_ptr<NeuropodTensorAllocator> allocator_;
    std::mutex                               allocator_lock_;

public:
    std::shared_ptr<NeuropodTensorAllocator> get_tensor_allocator()
    {
        std::lock_guard<std::mutex> lock(allocator_lock_);
        if (!allocator_)
        {
            allocator_ = std::make_shared<DefaultTensorAllocator<TensorImpl>>();
        }

        return allocator_;
    }
};

} // namespace neuropods
