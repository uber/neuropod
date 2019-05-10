project(tests)

# Begin gtest configuration
# Based on https://github.com/google/googletest/tree/master/googletest#using-cmake
# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# End gtest configuration

# Different definitions of origin
if(NOT APPLE)
  set(RPATH_ORIGIN "$ORIGIN")
else()
  set(RPATH_ORIGIN, "@executable_path")
endif()

# Build the tests
function(add_neuropod_test test_name additional_deps)
  add_executable( ${test_name} ${test_name}.cc)
  target_link_libraries(
    ${test_name} gtest gmock_main ${additional_deps}

    # Needed so the tests can find the backends to dynamically load
    "-Wl,-rpath,${RPATH_ORIGIN}/../backends/torchscript/"
    "-Wl,-rpath,${RPATH_ORIGIN}/../backends/tensorflow/"
    "-Wl,-rpath,${RPATH_ORIGIN}/../backends/python_bridge/"
  )
  add_test( ${test_name} ${test_name} )
endfunction()

enable_testing()


# Unit tests
add_neuropod_test(test_input_output neuropod_test_backend)
add_neuropod_test(test_config_utils neuropod_test_backend)
add_neuropod_test(test_internal_neuropod_tensor neuropod_test_backend)

# Integration tests with different backends
add_neuropod_test(test_python_bridge neuropod_pythonbridge_backend)
add_neuropod_test(test_torchscript_backend neuropod_torchscript_backend)
add_neuropod_test(test_tensorflow_backend neuropod_tensorflow_backend)
add_neuropod_test(test_default_backend neuropods)

# Eigen conversion test
find_package (Eigen3 REQUIRED NO_MODULE)
include_directories ( ${EIGEN3_INCLUDE_DIRS} )
add_neuropod_test(test_conversions_eigen neuropod_test_backend)
