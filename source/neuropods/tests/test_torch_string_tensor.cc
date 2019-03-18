//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"

#include "neuropods/backends/torchscript/torch_tensor.hh"

namespace
{

std::vector<std::string> get_sample_data(size_t numel)
{
    // Create a vector and fill it with sample data
    std::vector<std::string> sample_data;
    for (int i = 0; i < numel; i++) {
        sample_data.emplace_back("item" + std::to_string(i));
    }

    return sample_data;
}

}

TEST(test_torch_tensor, test_create_read_string_tensor)
{
    // Create a vector and fill it with sample data
    const std::vector<int64_t> dims = {1, 5, 6};
    auto numel = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
    const auto sample_data = get_sample_data(numel);

    // Convert to torchscript nested lists
    auto it = sample_data.begin();
    auto tensor = neuropods::make_nested_list(it, dims);

    // Get as a vector
    std::vector<std::string> out;
    out.reserve(numel);
    neuropods::row_major_fill(tensor, out);

    // Get the dims
    auto out_dims = neuropods::get_dims_from_nested_lists(tensor);

    // Check that everything matches
    EXPECT_TRUE(sample_data == out);
    EXPECT_TRUE(dims == out_dims);
}
