//
// Uber, Inc. (c) 2020
//

#include "neuropod/internal/tensor_types.hh"

namespace neuropod
{

// Used to print out the enum names rather than just a number
std::ostream &operator<<(std::ostream &out, const TensorType value)
{
    const char *s = 0;
#define GENERATE_CASE(item) \
    case (item):            \
        s = #item;          \
        break;
    switch (value)
    {
        GENERATE_CASE(FLOAT_TENSOR);
        GENERATE_CASE(DOUBLE_TENSOR);
        GENERATE_CASE(STRING_TENSOR);
        GENERATE_CASE(INT8_TENSOR);
        GENERATE_CASE(INT16_TENSOR);
        GENERATE_CASE(INT32_TENSOR);
        GENERATE_CASE(INT64_TENSOR);
        GENERATE_CASE(UINT8_TENSOR);
        GENERATE_CASE(UINT16_TENSOR);
        GENERATE_CASE(UINT32_TENSOR);
        GENERATE_CASE(UINT64_TENSOR);
    }
#undef GENERATE_CASE

    return out << s;
}

} // namespace neuropod
