project(python_bridge_backend)

# Find Python and Boost Python
find_package(PythonLibs 2.7 REQUIRED)
find_package(Boost 1.54 COMPONENTS python REQUIRED)

# Add the includes
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})

# Build the backend
file(GLOB SOURCES "*.cc")
add_library( neuropod_pythonbridge_backend SHARED ${SOURCES})
target_link_libraries(neuropod_pythonbridge_backend neuropods ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
