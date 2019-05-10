project(torchscript_backend)

# Build the backend
file(GLOB SOURCES "*.cc")
add_library( neuropod_torchscript_backend SHARED ${SOURCES})
target_link_libraries(neuropod_torchscript_backend neuropods ${TORCH_LIBRARIES})
