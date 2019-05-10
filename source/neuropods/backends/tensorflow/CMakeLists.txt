project(tensorflow_backend)

# Find the TF headers and shared object
find_path(TENSORFLOW_INCLUDE_DIR tensorflow/c/c_api.h)
find_library(LIBTENSORFLOW libtensorflow.so)

include_directories(${TENSORFLOW_INCLUDE_DIR})

# Build the backend
file(GLOB SOURCES "*.cc")
add_library( neuropod_tensorflow_backend SHARED ${SOURCES})
target_link_libraries(neuropod_tensorflow_backend ${LIBTENSORFLOW} neuropods)
