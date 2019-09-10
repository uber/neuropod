//
// Uber, Inc. (c) 2019
//

#include "neuropods/multiprocess/control_messages.hh"

namespace neuropods
{

// Used to print out the enum names rather than just a number
std::ostream& operator<<(std::ostream& out, const MessageType value)
{
    const char* s = 0;
#define GENERATE_CASE(item) case(item): s = #item; break;
    switch(value)
    {
        GENERATE_CASE(LOAD_NEUROPOD);
        GENERATE_CASE(ADD_INPUT);
        GENERATE_CASE(INFER);
        GENERATE_CASE(RETURN_OUTPUT);
        GENERATE_CASE(END_OUTPUT);
        GENERATE_CASE(INFER_COMPLETE);
        GENERATE_CASE(HEARTBEAT);
        GENERATE_CASE(SHUTDOWN);
    }
#undef GENERATE_CASE

    return out << s;
}

} // namespace neuropods
