//
// Uber, Inc. (c) 2019
//

#pragma once

namespace neuropod
{

constexpr int MAJOR_VERSION = 0;
constexpr int MINOR_VERSION = 1;
constexpr int PATCH_VERSION = 0;

static_assert(MINOR_VERSION < 100, "The minor version must be less than 100");
static_assert(PATCH_VERSION < 100, "The patch version must be less than 100");

constexpr int VERSION = (MAJOR_VERSION * 100 + MINOR_VERSION) * 100 + PATCH_VERSION;

} // namespace neuropod
