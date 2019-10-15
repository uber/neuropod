//
// Uber, Inc. (c) 2019
//

#include "gtest/gtest.h"

#include "neuropods/internal/backend_registration.hh"

namespace neuropods
{

TEST(test_backend_registration, invalid_name)
{
    EXPECT_THROW(get_backend_by_name("InvalidBackend"), std::runtime_error);
}

TEST(test_backend_registration, invalid_type)
{
    EXPECT_THROW(get_backend_for_type({}, "InvalidType"), std::runtime_error);
}

} // namespace neuropods
