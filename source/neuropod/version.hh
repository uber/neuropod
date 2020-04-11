//
// Uber, Inc. (c) 2019
//

#pragma once

// Inspired by patchlevel.h in cpython
namespace neuropod
{

// The values here must fit within 4 bits
#define NEUROPOD_RELEASE_LEVEL_NIGHTLY 0x5
#define NEUROPOD_RELEASE_LEVEL_ALPHA 0xA
#define NEUROPOD_RELEASE_LEVEL_BETA 0xB
#define NEUROPOD_RELEASE_LEVEL_RELEASE_CANDIDATE 0xC
#define NEUROPOD_RELEASE_LEVEL_FINAL_RELEASE 0xF

#define NEUROPOD_MAJOR_VERSION 0
#define NEUROPOD_MINOR_VERSION 2
#define NEUROPOD_PATCH_VERSION 0

// These are allowed to be 4 bits each
#define NEUROPOD_RELEASE_LEVEL NEUROPOD_RELEASE_LEVEL_RELEASE_CANDIDATE
#define NEUROPOD_RELEASE_SERIAL 1

static_assert(NEUROPOD_RELEASE_LEVEL < 16, "NEUROPOD_RELEASE_LEVEL must be in the range 0 to 15 (4 bits)");
static_assert(NEUROPOD_RELEASE_SERIAL < 16, "NEUROPOD_RELEASE_SERIAL must be in the range 0 to 15 (4 bits)");

// The version as a string
#define NEUROPOD_VERSION "0.2.0rc1"

// Version as a single 4-byte hex number, e.g. 0x010502B2 == 1.5.2b2.
// Use this for numeric comparisons
#define NEUROPOD_VERSION_HEX                                                                           \
    ((NEUROPOD_MAJOR_VERSION << 24) | (NEUROPOD_MINOR_VERSION << 16) | (NEUROPOD_PATCH_VERSION << 8) | \
     (NEUROPOD_RELEASE_LEVEL << 4) | (NEUROPOD_RELEASE_SERIAL << 0))

} // namespace neuropod
