//
// Uber, Inc. (c) 2019
//

#include "neuropod/multiprocess/control_messages.hh"

namespace neuropod
{

// Used to print out the enum names rather than just a number
std::ostream &operator<<(std::ostream &out, const MessageType value)
{
    const char *s = 0;
#define GENERATE_CASE(item) \
    case (item):            \
        s = #item;          \
        break;
    switch (value)
    {
        GENERATE_CASE(LOAD_NEUROPOD);
        GENERATE_CASE(LOAD_SUCCESS);
        GENERATE_CASE(ADD_INPUT);
        GENERATE_CASE(INFER);
        GENERATE_CASE(REQUEST_OUTPUT);
        GENERATE_CASE(RETURN_OUTPUT);
        GENERATE_CASE(END_OUTPUT);
        GENERATE_CASE(SHUTDOWN);
        GENERATE_CASE(EXCEPTION);
    }
#undef GENERATE_CASE

    return out << s;
}

} // namespace neuropod
