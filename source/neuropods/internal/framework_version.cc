//
// Uber, Inc. (c) 2019
//

#include "framework_version.hh"

#include "neuropods/internal/error_utils.hh"

#include <iostream>
#include <sstream>

namespace neuropods
{

// Supports simple parsing of version strings of the form MAJOR.MINOR.PATCH
FrameworkVersion parse_framework_version(const std::string &version)
{
    int major, minor, patch;
    if (sscanf(version.c_str(), "%d.%d.%d", &major, &minor, &patch) != 3)
    {
        NEUROPOD_ERROR("Error parsing version string. Expected format '<integer>.<integer>.<integer>' but got " << version);
    }

    return FrameworkVersion(major, minor, patch, "stable");
}

FrameworkVersion::FrameworkVersion(int major, int minor, int patch, const std::string &release_level, int serial)
    : major_(major), minor_(minor), patch_(patch), release_level_(release_level), serial_(serial)
    {}

FrameworkVersion::~FrameworkVersion() = default;

bool FrameworkVersion::does_match(const FrameworkVersion &other) const
{
    if (major_ != other.major_ && major_ >= 0 && other.major_ >= 0)
    {
        // Major version doesn't match
        return false;
    }

    if (minor_ != other.minor_ && minor_ >= 0 && other.minor_ >= 0)
    {
        // Minor version doesn't match
        return false;
    }

    if (patch_ != other.patch_ && patch_ >= 0 && other.patch_ >= 0)
    {
        // Patch version doesn't match
        return false;
    }

    if (release_level_ != other.release_level_ && !release_level_.empty() && !other.release_level_.empty())
    {
        // Release level doesn't match
        return false;
    }
    // Only look at serial if release matches
    else if (serial_ != other.serial_ && serial_ >= 0 && other.serial_ >= 0)
    {
        // Serial doesn't match
        return false;
    }

    return true;
}

std::ostream & operator << (std::ostream &out, const FrameworkVersion &f)
{
    out << f.major_ << "." << f.minor_ << "." << f.patch_;

    if (!f.release_level_.empty())
    {
        out << "." << f.release_level_ << f.serial_;
    }

    return out;
}

} // namespace neuropods
