project(test_backend)

# Build the backend
file(GLOB SOURCES "*.cc")
add_library( neuropod_test_backend SHARED ${SOURCES})
target_link_libraries(neuropod_test_backend neuropods)
