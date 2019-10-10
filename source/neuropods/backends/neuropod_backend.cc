//
// Uber, Inc. (c) 2019
//

#include "neuropods/backends/neuropod_backend.hh"
#include "neuropods/internal/neuropod_loader.hh"


namespace neuropods
{

NeuropodBackend::NeuropodBackend() = default;
NeuropodBackend::~NeuropodBackend() = default;

NeuropodBackend::NeuropodBackend(const std::string &neuropod_path)
{
    loader_ = get_loader(neuropod_path);
}

} // namespace neuropods
