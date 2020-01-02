//
// Uber, Inc. (c) 2019
//

#include "neuropods/internal/neuropod_loader.hh"

#include "neuropods/internal/error_utils.hh"
#include "neuropods/internal/memory_utils.hh"

#include <ghc/filesystem.hpp>

#include <fstream>
#include <iterator>
#include <sstream>

#include <picosha2.h>
#include <unzipper.h>

namespace neuropod
{

namespace
{

namespace fs = ghc::filesystem;

// Load a neuropod from a local directory on disk
class LocalLoader : public NeuropodLoader
{
private:
    std::string neuropod_path_;

public:
    LocalLoader(const std::string &neuropod_path) : neuropod_path_(neuropod_path) {}

    ~LocalLoader() = default;

    std::unique_ptr<std::istream> get_istream_for_file(const std::string &path)
    {
        return stdx::make_unique<std::ifstream>(get_file_path(path));
    }

    std::string get_file_path(const std::string &path)
    {
        // Sanity check for non relative paths
        // TODO(vip): Add more robust checking. This check is just to prevent accidentally
        // leading with a `/`.
        if (path.front() == '/')
        {
            NEUROPOD_ERROR("paths passed to get_file_path must be relative");
        }

        return fs::absolute(neuropod_path_) / path;
    }

    std::string ensure_local() { return neuropod_path_; }
};

// Loads a neuropod from a zipfile
class ZipLoader : public NeuropodLoader
{
private:
    zipper::Unzipper unzipper_;

    // Whether or not we unzipped the archive
    bool did_unzip_;

    // A temp dir that we unzipped to
    std::string tempdir_;

public:
    ZipLoader(const std::string neuropod_path) : unzipper_(neuropod_path), did_unzip_(false) {}

    ~ZipLoader()
    {
        if (did_unzip_)
        {
            // Delete the folder
            fs::remove_all(tempdir_);
        }
    }

    std::unique_ptr<std::istream> get_istream_for_file(const std::string &path)
    {
        auto out = stdx::make_unique<std::stringstream>();
        unzipper_.extractEntryToStream(path, *out);
        return out;
    }

    std::string get_file_path(const std::string &path)
    {
        // Sanity check for non relative paths
        // TODO(vip): Add more robust checking. This check is just to prevent accidentally
        // leading with a `/`.
        if (path.front() == '/')
        {
            NEUROPOD_ERROR("paths passed to get_file_path must be relative");
        }

        if (did_unzip_)
        {
            return fs::absolute(tempdir_) / path;
        }
        else
        {
            // TODO(vip): only extract the requested file
            // This extracts the entire archive
            ensure_local();
            return get_file_path(path);
        }
    }

    std::string ensure_local()
    {
        // Create a tempdir
        char tempdir[] = "/tmp/neuropod_tmp_XXXXXX";
        if (mkdtemp(tempdir) == nullptr)
        {
            NEUROPOD_ERROR("Error creating temporary directory")
        }

        // Unzip into the tempdir
        unzipper_.extract(tempdir);

        // Update metadata to make sure we cleanup
        did_unzip_ = true;
        tempdir_   = tempdir;

        return tempdir;
    }
};

} // namespace

NeuropodLoader::~NeuropodLoader() = default;

// Get the SHA256 of a file
std::string NeuropodLoader::get_hash_for_file(const std::string &path)
{
    auto                       stream = get_istream_for_file(path);
    std::vector<unsigned char> hash(picosha2::k_digest_size);
    picosha2::hash256(
        std::istreambuf_iterator<char>(*stream), std::istreambuf_iterator<char>(), hash.begin(), hash.end());

    return picosha2::bytes_to_hex_string(hash.begin(), hash.end());
}

// Get a loader given a path to a file or directory.
// If this is a file, it is assumed to be a zipfile containing a neuropod
std::unique_ptr<NeuropodLoader> get_loader(const std::string &neuropod_path)
{
    if (!fs::exists(neuropod_path))
    {
        NEUROPOD_ERROR("Error loading Neuropod. No file or directory at '" << neuropod_path << "'");
    }

    if (fs::is_directory(neuropod_path))
    {
        return stdx::make_unique<LocalLoader>(neuropod_path);
    }
    else
    {
        return stdx::make_unique<ZipLoader>(neuropod_path);
    }
}

} // namespace neuropod
