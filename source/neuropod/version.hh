/* Copyright (c) 2020 The Neuropod Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
#define NEUROPOD_MINOR_VERSION 3
#define NEUROPOD_PATCH_VERSION 0

// These are allowed to be 4 bits each
#define NEUROPOD_RELEASE_LEVEL NEUROPOD_RELEASE_LEVEL_RELEASE_CANDIDATE
#define NEUROPOD_RELEASE_SERIAL 7

static_assert(NEUROPOD_RELEASE_LEVEL < 16, "NEUROPOD_RELEASE_LEVEL must be in the range 0 to 15 (4 bits)");
static_assert(NEUROPOD_RELEASE_SERIAL < 16, "NEUROPOD_RELEASE_SERIAL must be in the range 0 to 15 (4 bits)");

// The version as a string
#define NEUROPOD_VERSION "0.3.0rc7"

// Version as a single 4-byte hex number, e.g. 0x010502B2 == 1.5.2b2.
// Use this for numeric comparisons
#define NEUROPOD_VERSION_HEX                                                                           \
    ((NEUROPOD_MAJOR_VERSION << 24) | (NEUROPOD_MINOR_VERSION << 16) | (NEUROPOD_PATCH_VERSION << 8) | \
     (NEUROPOD_RELEASE_LEVEL << 4) | (NEUROPOD_RELEASE_SERIAL << 0))

} // namespace neuropod
