//
// Uber, Inc. (c) 2019
//

#pragma once

#include <string>
#include <iostream>
#include <memory>

namespace neuropods
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

    // If this is a zipped neuropod, extract to a temp dir and return the extracted path
    // Otherwise, return the neuropod_path
    virtual std::string ensure_local() = 0;
};

// Get a loader given a path to a file or directory.
// If this is a file, it is assumed to be a zipfile containing a neuropod
std::unique_ptr<NeuropodLoader> get_loader(const std::string &neuropod_path);

} // namespace neuropods
