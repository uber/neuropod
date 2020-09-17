/* Copyright (c) 2020 UATC, LLC

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

#include "neuropod/multiprocess/control_messages.hh"

namespace neuropod
{

// Used to print out the enum names rather than just a number
std::ostream &operator<<(std::ostream &out, const MessageType value)
{
    const char *s = nullptr;
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
        GENERATE_CASE(RETURN_OUTPUT);
        GENERATE_CASE(SHUTDOWN);
        GENERATE_CASE(EXCEPTION);
    }
#undef GENERATE_CASE

    return out << s;
}

} // namespace neuropod
