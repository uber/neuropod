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

#include "neuropod/multiprocess/multiprocess_worker.hh"

#include <iostream>
#include <string>

// A worker process that runs a neuropod
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::string program_name(argv[0]);
        std::cout << "Usage: " + program_name + " control_queue_name" << std::endl;
        return 1;
    }

    std::string control_queue_name(argv[1]);

    // Start the main loop
    neuropod::multiprocess_worker_loop(control_queue_name);
}
