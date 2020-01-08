# Building Neuropod

Neuropod uses [Bazel](https://bazel.build/) as a build system.

There are a few ways to build the project:

- Natively on Linux and Mac
- In Docker (preferred)

## Natively on Linux and Mac

The following sets up a local environment for building and testing:

```sh
# Do everything in a virtualenv (optional)
sudo pip install -U pip
sudo pip install virtualenv
virtualenv /tmp/neuropod_venv
source /tmp/neuropod_venv/bin/activate

# Install system dependencies (e.g. bazel)
./build/install_system_deps.sh

# Install python dependencies (e.g. numpy)
./build/install_python_deps.sh
```

After the above steps, you can run the following scripts:

```sh
# Note: If you used a virtualenv above, make sure it's still activated
# during these steps

# Build
./build/build.sh

# Run tests
./build/test.sh
# Or ./build/test_gpu.sh to run all tests
```

## In Docker (preferred)

```sh
./build/docker_build.sh
```

Internally, this uses the build scripts mentioned above, but provides better isolation between builds.

Also, compared to a naive docker build, this command preserves the bazel cache. This ensures that subsequent builds run as quickly as possible.

### Debugging/interactively building

In order to debug and/or experiment, it may be useful to build interactively within Docker:

```sh
# Will launch bash in a docker container containing all Neuropod dependencies
./build/docker_build.sh -i

# Run these commands inside the container in order to build and test
./build/build.sh
./build/test.sh
```

## Tests

Neuropod has a set of tests implemented in C++ and a set of tests implemented in Python. Test coverage is described below:

| | Location | Covers C++ Library | Covers Python Library |
| --- | --- | :---: | :---: |
| C++ Tests | `source/neuropod/tests/*` | x |  |
| Python Tests | `source/neuropod/python/tests/test_*` | x | x |
| GPU-only Python Tests | `source/neuropod/python/tests/gpu_test_*` | x | x |

The Python tests run against both the Python and C++ libraries by using python bindings. This means that many tests only need to be written in Python.

## CI

### Build Matrix

Our build matrix is defined as all combinations of the following:

**Platform:**

 - Ubuntu 16.04 GPU (in Docker) - Buildkite
 - Ubuntu 16.04 CPU (in Docker) - Buildkite
 - Mac CPU (native) - Travis CI

**Python:**

 - 2.7
 - 3

**Framework versions (each row of the table):**

| CUDA | TF | Torch |
| --- | --- | --- |
| 9.0 | 1.12.0 | 1.1.0 |
| 10.0 | 1.13.1 | 1.2.0 |
| 10.0 | 1.14.0 | 1.3.0.dev20190820 |
| 10.0 | 1.15.0 | 1.3.0 |

We also have an additional build that runs natively on an Ubuntu 16.04 machine (outside of docker)
to make sure that we don't accidentally break workflows of users not using docker.

This is a total of 25 builds (4 * 2 * 3 + 1) running in CI

The current build matrix is defined in [build/ci_matrix.py](https://github.com/uber/neuropods/blob/master/build/ci_matrix.py#L73-L91)

### Future additions

We're also planning on adding the following configurations to the build matrix:

**Configs:**

 - ASAN + Code Coverage
 - Release
