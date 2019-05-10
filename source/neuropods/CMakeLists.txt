cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(neuropods)
set (CMAKE_CXX_STANDARD 11)

# The header include paths are relative to the parent directory
# (i.e. they are of the form `neuropods/...`)
include_directories("${CMAKE_SOURCE_DIR}/../")

if (NOT APPLE)
  # Make sure all symbols are defined in all the libraries we build
  set (CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

  # Link libraries we specify even if there is no compile-time dependency
  # (this is needed in the tests)
  set (CMAKE_EXE_LINKER_FLAGS "-Wl,--no-as-needed")
endif()

# Load libtorch. This needs to happen here so both the tests and the
# backends can find the library
# TODO(vip): refactor so this doesn't need to be in the root CMakeLists.txt
find_package(Torch REQUIRED)

# Add the backends
add_subdirectory(backends)

# Build the main library
file(GLOB INTERNAL_SOURCES "internal/*.cc")
add_library( neuropods SHARED ${INTERNAL_SOURCES} "neuropods.cc")
target_link_libraries(neuropods dl jsoncpp) 

# Add the tests
add_subdirectory(tests)
