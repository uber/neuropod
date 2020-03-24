//
// Uber, Inc. (c) 2018
//

#pragma once

#include "neuropod/internal/deleter.hh"
#include "neuropod/internal/neuropod_tensor.hh"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace neuropod
{

// This is used along with the TestNeuropodBackend in tests
template <typename T>
class TestNeuropodTensor : public TypedNeuropodTensor<T>
{
protected:
    // A pointer to the data contained in the tensor
    std::shared_ptr<void> data_;

public:
    TestNeuropodTensor(const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<T>(dims), data_(malloc(this->get_num_elements() * sizeof(T)), free)
    {
    }

    // Wrap existing memory
    TestNeuropodTensor(const std::vector<int64_t> &dims, void *data, const Deleter &deleter)
        : TypedNeuropodTensor<T>(dims), data_(data, deleter)
    {
    }

    ~TestNeuropodTensor() {}

protected:
    // Get a pointer to the underlying data
    void *get_untyped_data_ptr() { return data_.get(); }

    const void *get_untyped_data_ptr() const { return data_.get(); }

    // TestNeuropodTensor should not be used for inference so it's okay to return a nullptr here
    std::shared_ptr<NeuropodValue> seal(NeuropodDevice device) { return nullptr; }
};

// Specialization for strings
template <>
class TestNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>
{
protected:
    // The data contained in the tensor
    std::vector<std::string> data_;

public:
    TestNeuropodTensor(const std::vector<int64_t> &dims) : TypedNeuropodTensor<std::string>(dims) {}

    ~TestNeuropodTensor() = default;

    void set(const std::vector<std::string> &data) { data_ = data; }

protected:
    const std::string operator[](size_t index) const { return data_[index]; }

    // TestNeuropodTensor should not be used for inference so it's okay to return a nullptr here
    std::shared_ptr<NeuropodValue> seal(NeuropodDevice device) { return nullptr; }
};

} // namespace neuropod
