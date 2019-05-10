project(neuropod_backends)

# Add each of the backends
add_subdirectory(python_bridge)
add_subdirectory(tensorflow)
add_subdirectory(test_backend)
add_subdirectory(torchscript)
