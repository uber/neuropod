//
// Uber, Inc. (c) 2019
//

#include <iostream>
#include <string>
#include <vector>

#include "neuropods/neuropods.hh"
#include "neuropods/backends/multiprocess_backend/shm_tensor.hh"
#include "neuropods/internal/tensor_utils.hh"

// A worker process that runs a neuropod
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::string program_name(argv[0]);
        std::cout << "Usage: " + program_name + " neuropod_path" << std::endl;
        return 1;
    }

    // Load the specified neuropod
    std::string neuropod_path(argv[1]);
    neuropods::Neuropod neuropod(neuropod_path);

    std::string line;
    std::unordered_set<std::shared_ptr<neuropods::NeuropodTensor>> inputs;

    // The last outputs
    // We need to keep these around so there isn't a race condition when returning
    // data back to the main process
    std::unordered_set<std::shared_ptr<neuropods::NeuropodTensor>> last_outputs;

    // Let the main process know that we're ready
    std::cout << "ready" << std::endl;
    while (std::getline(std::cin, line)) {
        if (line == "infer")
        {
            // Run inference and then empty our input set
            auto outputs = neuropod.infer(inputs);
            inputs.clear();

            // Print out the shm keys of the outputs
            for (const auto &tensor : outputs->tensors)
            {
                // Turn this "native" tensor into an shm tensor
                // Unfortunately, this requires a copy (done within SHMNeuropodTensor)
                auto shm_tensor = neuropods::wrap_existing_tensor<neuropods::SHMNeuropodTensor>(tensor);

                // This ensures that the tensor stays around long enough for the other process to load it
                last_outputs.insert(shm_tensor);

                // Get the shm key
                const auto &shm_key
                    = std::dynamic_pointer_cast<neuropods::NativeDataContainer<std::string>>(shm_tensor)->get_native_data();

                // Send it to the main process
                std::cout << shm_key << std::endl;
            }

            // Let the main process know that we're done
            std::cout << "end_output" << std::endl;
        }
        else if (line == "infer_complete")
        {
            // The other process loaded our output tensors
            // Empty our last set of outputs
            last_outputs.clear();
        }
        else
        {
            // We're adding an input
            // Load an shm tensor based on the key sent from the main process
            auto shm_tensor = neuropods::tensor_from_shm_key(line);

            // Create a "native" tensor with the data in the shm tensor
            // This doesn't do a copy; it just wraps the data and passes it to the
            // underlying backend
            auto tensor = neuropods::wrap_existing_tensor(neuropod, shm_tensor);

            // Add it to the inputs
            inputs.insert(tensor);
        }
    }
}
