//
// Uber, Inc. (c) 2018
//

#pragma once

namespace neuropods
{

class Neuropod;

// The following types are opaque to the user
class NeuropodBackend;

struct TensorStore;

struct NeuropodInputData;
struct NeuropodInputDataDeleter
{
    void operator()(NeuropodInputData *p);
};

} // namespace neuropods