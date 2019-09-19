//
// Uber, Inc. (c) 2019
//

#pragma once

#include <string>

namespace neuropods
{

// A struct to represent the version of a framework
class FrameworkVersion
{
private:
    // The major version (e.g. 1)
    int major_;

    // The minor version (e.g. 0)
    int minor_;

    // The patch version (e.g. 0)
    int patch_;

    // The release level (e.g. "stable", "nightly", "alpha")
    std::string release_level_;

    // The serial (e.g. 20190918 for a nightly release)
    int serial_;

public:
    // The default values match with any framework version
    FrameworkVersion(int major = -1, int minor = -1, int patch = -1, const std::string &release_level = "", int serial = -1);

    ~FrameworkVersion();

    // Returns whether this FrameworkVersion matches another one (ignoring empty fields and ones < 0)
    bool does_match(const FrameworkVersion &other) const;

    // To pretty print FrameworkVersion objects
    friend std::ostream & operator << (std::ostream &out, const FrameworkVersion &f);
};

// Supports simple parsing of version strings of the form MAJOR.MINOR.PATCH
FrameworkVersion parse_framework_version(const std::string &version);

} // namespace neuropods
