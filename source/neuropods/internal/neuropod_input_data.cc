//
// Uber, Inc. (c) 2018
//

#include "neuropod_input_data.hh"

namespace neuropods
{

void NeuropodInputDataDeleter::operator()(NeuropodInputData *p)
{
    delete p;
}

} // namespace neuropods