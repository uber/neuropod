//
// Uber, Inc. (c) 2018
//

#pragma once

#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "neuropods/internal/neuropod_tensor.hh"

namespace neuropods
{

namespace
{

template <typename T>
T *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    return tensor.data<T>();
}

template <>
uint16_t *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    throw std::runtime_error("TorchScript doesn't support type uint16_t");
}

template <>
uint32_t *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    throw std::runtime_error("TorchScript doesn't support type uint32_t");
}

template <>
uint64_t *get_data_from_torch_tensor(const torch::Tensor &tensor)
{
    throw std::runtime_error("TorchScript doesn't support type uint64_t");
}

} // namespace

// This class is internal to neuropods and should not be exposed
// to users
template <typename T>
class TorchNeuropodTensor : public TypedNeuropodTensor<T>, public NativeDataContainer<torch::jit::IValue>
{
public:
    // Allocate a torch tensor
    // TODO(vip): maybe add a way to wrap existing data using torch::from_blob
    TorchNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<T>(name, dims),
          tensor(torch::empty(dims, get_torch_type_from_neuropod_type(get_tensor_type_from_cpp<T>())))
    {
    }

    // Wrap an existing torch tensor
    TorchNeuropodTensor(const std::string &name, torch::Tensor tensor)
        : TypedNeuropodTensor<T>(name, tensor.sizes().vec()), tensor(tensor)
    {
    }

    ~TorchNeuropodTensor() = default;

    // Get a pointer to the underlying data
    T *get_raw_data_ptr() { return get_data_from_torch_tensor<T>(tensor); }

    // Get a pointer to the underlying data
    const T *get_raw_data_ptr() const { return get_data_from_torch_tensor<T>(tensor); }

    torch::jit::IValue get_native_data() { return tensor; }

    // The underlying torch tensor
    torch::Tensor tensor;
};

// Utility function to get the dims of a string "tensor" represented by nested
// lists.
// Note: this function assumes that the list of lists are rectangular (i.e. the
// level of nesting is the same throughout the whole structure)
std::vector<int64_t> get_dims_from_nested_lists(const c10::intrusive_ptr<at::ivalue::GenericList> &input)
{
    std::vector<int64_t> out;

    c10::intrusive_ptr<at::ivalue::GenericList> curr = input;
    while (true)
    {
        out.push_back(curr->elements().size());

        if (curr->elements()[0].isGenericList())
        {
            // List of lists meaning that there are more dims
            curr = curr->elements()[0].toGenericList();
        }
        else
        {
            // No more nested lists
            return out;
        }
    }
}

// Iterates through nested lists in row major order and fills a vector of output data
void row_major_fill(const c10::intrusive_ptr<at::ivalue::GenericList> &input, std::vector<std::string> &output_data)
{
    const auto &elems = input->elements();
    if (elems[0].isGenericList())
    {
        for (const auto &elem : elems)
        {
            // Recursive call
            row_major_fill(elem.toGenericList(), output_data);
        }
    }
    else
    {
        // Copy the data into the vector
        for (const auto &elem : elems)
        {
            output_data.push_back(elem.toStringRef());
        }
    }
}

// Iterates through a vector and creates nested lists to match the specified dimensions
c10::intrusive_ptr<at::ivalue::GenericList> make_nested_list(std::vector<std::string>::const_iterator &it,
                                                             const std::vector<int64_t> &              dims,
                                                             int                                       depth = 0)
{
    std::vector<torch::jit::IValue> out;

    const int dim_size = dims[depth];
    if (depth == dims.size() - 1)
    {
        // Base case
        for (int i = 0; i < dim_size; i++)
        {
            // Get the next item from the iterator
            // Note: there is a bounds check that happens outside of this function
            // so we don't need to repeat it in this loop
            out.emplace_back(*(it++));
        }
    }
    else
    {
        // Recursive call
        for (int i = 0; i < dim_size; i++)
        {
            out.emplace_back(make_nested_list(it, dims, depth + 1));
        }
    }

    return at::ivalue::GenericList::create(out);
}

// Specialization for strings
// Torch does not natively support string tensors. We will be using
// (potentially nested) lists instead
template <>
class TorchNeuropodTensor<std::string> : public TypedNeuropodTensor<std::string>,
                                         public NativeDataContainer<torch::jit::IValue>
{
public:
    // Allocate a torch tensor
    TorchNeuropodTensor(const std::string &name, const std::vector<int64_t> &dims)
        : TypedNeuropodTensor<std::string>(name, dims)
    {
    }

    // Wrap an existing string "tensor"
    TorchNeuropodTensor(const std::string &name, torch::jit::IValue tensor)
        : TypedNeuropodTensor<std::string>(name, get_dims_from_nested_lists(tensor.toGenericList())),
          list_(tensor.toGenericList())
    {
    }

    ~TorchNeuropodTensor() = default;

    void set(const std::vector<std::string> &data)
    {
        // Sanity check
        if (data.size() != get_num_elements())
        {
            throw std::runtime_error(
                "Error. Size of supplied vector does not match the number of elements in the tensor.");
        }

        // Get a const_iterator from the vector
        auto it = data.begin();

        // Make the list
        list_ = make_nested_list(it, get_dims());
    }

    std::vector<std::string> get_data_as_vector()
    {
        std::vector<std::string> out;
        // Reserve space for the whole tensor
        out.reserve(get_num_elements());

        // Fill the vector in row major order
        row_major_fill(list_, out);

        // Sanity check
        if (out.size() != get_num_elements())
        {
            throw std::runtime_error("Error converting TorchScript list into vector of strings. "
                                     "Make sure that the dimensions of the returned list are correct.");
        }

        // Return the filled vector
        return out;
    }

    torch::jit::IValue get_native_data() { return list_; }

    // The underlying torchscript list
    c10::intrusive_ptr<at::ivalue::GenericList> list_;
};

} // namespace neuropods
