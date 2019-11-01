There are a handful of ways of creating tensors from exisiting data in C++ and they all have different tradeoffs between simplicity and performance. This document goes over some approaches and their tradeoffs.

!!! tip
    Make sure to read the C++ guide before continuing

## Writing your data directly into a tensor

This is preferable if you can do it.

Instead of copying data or wrapping existing data, write your data directly into a tensor.

### Pros

- This'll work without copies under the hood for both in-process and out-of-process execution
- Has no memory alignment requirements
- No need to work with deleters

### Cons

- It can require a lot of refactoring of an existing application in order to make this work well

### Examples

You could receive data directly into a tensor's underlying buffer:
```cpp
// Allocate a tensor
auto tensor = allocator->allocate_tensor<float>({6, 6});

// Get a pointer to the underlying buffer
auto data = tensor->get_raw_data_ptr();

// Some function that writes data directly into this buffer
recv_message_into_buffer(data);
```


Or you could manually fill in a tensor:
```cpp
// Allocate a tensor
auto tensor = allocator->allocate_tensor<float>({256, 256});
const auto &dims = tensor->get_dims();

// Get an accessor
auto accessor = tensor->accessor<2>();

// Write data directly into it
for (int i = 0; i < dims[0]; i++)
{
    for (int j = 0; j < dims[1]; j++)
    {
        accessor[i][j] = i * j;
    }
}
```

You could even parallelize that with TBB:
```cpp
// Allocate a tensor
auto tensor = allocator->allocate_tensor<float>({256, 256});
const auto &dims = tensor->get_dims();

// Get an accessor
auto accessor = tensor->accessor<2>();

// Write data into the tensor in parallel
tbb::parallel_for(
    // Parallelize in blocks of 16 by 16
    tbb:blocked_range2d<size_t>(0, dims[0], 16, 0, dims[1], 16),

    // Run this lambda in parallel for each block in the range above
    [&](const blocked_range2d<size_t>& r) {
        for(size_t i = r.rows().begin(); i != r.rows().end(); i++)
        {
            for(size_t j = r.cols().begin(); j != r.cols().end(); j++)
            {
                accessor[i][j] = i * j;
            }
        }
    }
);

```

## Wrapping existing memory

This works well if you already have your data in a buffer somewhere.

### Pros

- This'll work without copies during in-process execution
- Easy to do if you already have data

### Cons

- Need an understanding of what deleters are and how to use them correctly
- For efficient usage with TF, the data needs to be 64 byte aligned
    - Note: this isn't a hard requirement, but TF may copy unaligned data under the hood
- Compared to #1, this makes an extra copy during out-of-process execution

### Examples

Wrapping data from a `cv::Mat`:
```cpp
cv::Mat image = ... // An image from somewhere
auto tensor = allocator->tensor_from_memory<uint8_t>(
    // Dimensions
    {1, image.rows, image.cols, image.channels()},

    // Data
    image.data,

    // Deleter
    [image](void * unused) {
        // By capturing `image` in this deleter, we ensure
        // that the underlying data does not get deallocated
        // before we're done with the tensor.
    }
);
```

## Copying data into a tensor

### Pros

- Very easy to do
- No memory alignment requirements
- No need to work with deleters

### Cons

- Always makes an extra copy during in-process execution
- Compared to #1, this makes an extra copy during out-of-process execution (although this copy is explicitly written by the user)

### Examples

Copying from a `cv::Mat`:
```cpp
cv::Mat image = ... // An image from somewhere
auto tensor = allocator->allocate_tensor<uint8_t>(
    // Dimensions
    {1, image.rows, image.cols, image.channels()}
);

// Copy data into the tensor
tensor->copy_from(image.data, tensor->get_num_elements());
```


## Which one should I use?

In general, the order of approaches in terms of performance is the following:

1. Writing data directly into a tensor
2. Wrapping existing memory
3. Copying data into a tensor

That said, profiling is your friend.

The tradeoff between simplicity and performance is also different for large vs small tensors since copies are cheaper for small tensors.
