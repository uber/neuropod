//
// Uber, Inc. (c) 2020
//

#pragma once

#include "neuropod/backends/test_backend/test_neuropod_tensor.hh"
#include "neuropod/bindings/python_bindings.hh"

namespace neuropod
{

class SealedPythonTensor : public SealedNeuropodTensor
{
public:
    std::unique_ptr<py::array> arr_;

    SealedPythonTensor(py::array arr) : arr_(stdx::make_unique<py::array>(std::move(arr))) {}
    ~SealedPythonTensor()
    {
        // Acquire the GIL
        py::gil_scoped_acquire gil;

        // Destroy the array
        arr_.reset();
    }
};

template <typename T>
class PythonNeuropodTensor : public TestNeuropodTensor<T>
{
public:
    // Allocate memory
    PythonNeuropodTensor(const std::vector<int64_t> &dims) : TestNeuropodTensor<T>(dims) {}

    // Wrap existing memory
    PythonNeuropodTensor(const std::vector<int64_t> &dims, void *data, const Deleter &deleter)
        : TestNeuropodTensor<T>(dims, data, deleter)
    {
    }

    ~PythonNeuropodTensor() = default;

protected:
    std::shared_ptr<SealedNeuropodTensor> seal(NeuropodDevice device)
    {
        // Acquire the GIL
        py::gil_scoped_acquire gil;

        // Make the SealedPythonTensor
        auto &data = this->data_;
        return std::make_shared<SealedPythonTensor>(tensor_to_numpy(*this, [data](void * /* unused */) {}));
    }
};

// Specialization for strings
template <>
class PythonNeuropodTensor<std::string> : public TestNeuropodTensor<std::string>
{
public:
    PythonNeuropodTensor(const std::vector<int64_t> &dims) : TestNeuropodTensor<std::string>(dims) {}

    ~PythonNeuropodTensor() = default;

protected:
    std::shared_ptr<SealedNeuropodTensor> seal(NeuropodDevice device)
    {
        // Acquire the GIL
        py::gil_scoped_acquire gil;

        // Make the SealedPythonTensor
        return std::make_shared<SealedPythonTensor>(tensor_to_numpy(*this, [](void * /* unused */) {}));
    }
};

} // namespace neuropod
