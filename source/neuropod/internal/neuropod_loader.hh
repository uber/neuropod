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

#include <iostream>
#include <memory>
#include <string>

namespace neuropod
{

class NeuropodLoader
{
public:
    virtual ~NeuropodLoader();

    // Get an istream given a relative path in the loaded neuropod
    virtual std::unique_ptr<std::istream> get_istream_for_file(const std::string &path) = 0;

    // Gets a path to a file within a neuropod
    // If this is a zipped neuropod, this will extract that file to a temp
    // dir and return the path. Otherwise it'll just return the full path to that
    // file
    virtual std::string get_file_path(const std::string &path) = 0;

    // Get the SHA256 of a file
    std::string get_hash_for_file(const std::string &path);

    // If this is a zipped neuropod, extract to a temp dir and return the extracted path
    // Otherwise, return the neuropod_path
    virtual std::string ensure_local() = 0;
};

// Get a loader given a path to a file or directory.
// If this is a file, it is assumed to be a zipfile containing a neuropod
std::unique_ptr<NeuropodLoader> get_loader(const std::string &neuropod_path);

} // namespace neuropod
