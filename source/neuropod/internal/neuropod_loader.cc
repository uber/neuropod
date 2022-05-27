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

#include "neuropod/internal/neuropod_loader.hh"

#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/memory_utils.hh"

#include <ghc/filesystem.hpp>

#include <fstream>
#include <iterator>
#include <sstream>

#include <mz.h>
#include <mz_strm.h>
#include <mz_strm_os.h>
#include <mz_zip.h>
#include <mz_zip_rw.h>
#include <picosha2.h>

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
    explicit LocalLoader(std::string neuropod_path) : neuropod_path_(std::move(neuropod_path)) {}

    ~LocalLoader() override = default;

    std::unique_ptr<std::istream> get_istream_for_file(const std::string &path) override
    {
        auto ret = stdx::make_unique<std::ifstream>(get_file_path(path));
        if (!(*ret))
        {
            return nullptr;
        }

        return ret;
    }

    std::string get_file_path(const std::string &path) override
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

    std::string ensure_local() override { return neuropod_path_; }
};

// Loads a neuropod from a zipfile
class ZipLoader : public NeuropodLoader
{
private:
    void *zip_reader_;

    // Whether or not we unzipped the archive
    bool did_unzip_;

    // A temp dir that we unzipped to
    std::string tempdir_;

public:
    explicit ZipLoader(const std::string neuropod_path) : did_unzip_(false)
    {
        // Create a zip reader and open the file
        mz_zip_reader_create(&zip_reader_);
        auto err = mz_zip_reader_open_file(zip_reader_, neuropod_path.c_str());
        if (err != MZ_OK)
        {
            NEUROPOD_ERROR("Error opening zip stream for neuropod model at {}", neuropod_path);
        }
    }

    ~ZipLoader() override
    {
        mz_zip_reader_close(zip_reader_);
        mz_zip_reader_delete(&zip_reader_);

        if (did_unzip_)
        {
            // Delete the folder
            fs::remove_all(tempdir_);
        }
    }

    std::unique_ptr<std::istream> get_istream_for_file(const std::string &path) override
    {
        auto out = stdx::make_unique<std::stringstream>();

        // Find the file we're looking for
        auto err = mz_zip_reader_locate_entry(zip_reader_, path.c_str(), 1);
        if (err != MZ_OK)
        {
            SPDLOG_TRACE("Couldn't find file in zip: {}", path);
            return nullptr;
        }

        // Open it
        err = mz_zip_reader_entry_open(zip_reader_);
        if (err != MZ_OK)
        {
            SPDLOG_TRACE("Couldn't open file '{}' in zip file: {}", path, err);
            return nullptr;
        }

        char    buf[4096];
        int32_t bytes_read = 0;

        do
        {
            // Read chunks and write to the output stream
            bytes_read = mz_zip_reader_entry_read(zip_reader_, buf, sizeof(buf));
            if (bytes_read < 0)
            {
                err = bytes_read;
                break;
            }

            out->write(buf, bytes_read);
        } while (bytes_read > 0);

        // Close the reader
        mz_zip_reader_entry_close(zip_reader_);

        // Make sure we correctly read the file
        if (err != MZ_OK)
        {
            SPDLOG_TRACE("Error reading zip file entry: {}", err);
            return nullptr;
        }

        return out;
    }

    std::string get_file_path(const std::string &path) override
    {
        // Sanity check for non relative paths
        // TODO(vip): Add more robust checking. This check is just to prevent accidentally
        // leading with a `/`.
        if (path.front() == '/')
        {
            NEUROPOD_ERROR("paths passed to get_file_path must be relative");
        }

        if (!did_unzip_)
        {
            // TODO(vip): only extract the requested file
            // This extracts the entire archive
            ensure_local();
        }

        return fs::absolute(tempdir_) / path;
    }

    std::string ensure_local() override
    {
        // Create a tempdir
        char tempdir[] = "/tmp/neuropod_tmp_XXXXXX";
        if (mkdtemp(tempdir) == nullptr)
        {
            NEUROPOD_ERROR("Error creating temporary directory");
        }

        // Unzip into the tempdir
        if (mz_zip_reader_save_all(zip_reader_, tempdir) != MZ_OK)
        {
            NEUROPOD_ERROR("Error unzipping neuropod model");
        }

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
        NEUROPOD_ERROR("Error loading Neuropod. No file or directory at '{}'", neuropod_path);
    }

    if (fs::is_directory(neuropod_path))
    {
        return stdx::make_unique<LocalLoader>(neuropod_path);
    }

    return stdx::make_unique<ZipLoader>(neuropod_path);
}

} // namespace neuropod
