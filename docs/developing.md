# Building Neuropod

Neuropod uses [Bazel](https://bazel.build/) as a build system.

There are a few ways to build the project:

- Natively on Linux and Mac
- In Docker (preferred)

## Natively on Linux and Mac

The following sets up a local environment for building and testing:

```sh
# Install system dependencies (e.g. bazel)
./build/install_system_deps.sh

# Install python dependencies (e.g. numpy)
# Note: This creates a virtualenv for neuropod and installs all deps in it
./build/install_python_deps.sh
```

After the above steps, you can run the following scripts:

```sh
# Build
./build/build.sh

# Run tests
./build/test.sh
# Or ./build/test_gpu.sh to run all tests
```

!!! note
    The above commands run all python code in the virtualenv created by `install_python_deps.sh`. You do not need to manually activate the virtualenv.

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
| Python Tests | `source/python/neuropod/tests/test_*` | x | x |
| GPU-only Python Tests | `source/python/neuropod/tests/gpu_test_*` | x | x |

The Python tests run against both the Python and C++ libraries by using python bindings. This means that many tests only need to be written in Python.

C++ tests can have the following tags:

 - `gpu`: Only run this test when running GPU tests
 - `requires_ld_library_path`: Set the `LD_LIBRARY_PATH` and `PATH` environment variables so the backends and multiprocess worker are available. This is useful for tests that run a model using OPE.
 - `no_trace_logging`: Don't set the log level to `TRACE` when running this test. This is useful to avoid lots of output when running benchmarks.

## CI

### Build Matrix

Our build matrix is defined as all combinations of the following:

**Platform:**

 - Ubuntu 16.04 GPU (in Docker) - Buildkite
 - Ubuntu 16.04 CPU (in Docker) - Buildkite
 - Mac CPU (native) - GitHub Actions

**Framework versions (each row of the table):**

| CUDA | TF | Torch | Python |
| --- | --- | --- | --- |
| 9.0 | 1.12.0 | 1.1.0 | 2.7 |
| 10.0 | 1.13.1 | 1.2.0 | 3.5 |
| 10.0 | 1.14.0 | 1.3.0 | 3.6 |
| 10.0 | 1.15.0 | 1.4.0 | 3.7 |
| 10.1 | 2.2.0 | 1.5.0 | 3.8 |
| 10.1 | - | 1.6.0 | 3.8 |
| 10.1 | - | 1.7.0 | 3.8 |
| 11.2.1 | 2.5.0 | - | 3.8 |
| 11.2.1 | 2.6.2 | - | 3.8 |

We also have the following ad-hoc builds:

 - A lint + docs + static analysis build (Buildkite)
 - A native Ubuntu 16.04 build (outside of docker) to make sure that we don't accidentally break workflows of users not using docker

This is a total of 17 builds (3 * 5 + 2) running in CI

The current build matrix is defined in [build/ci_matrix.py](https://github.com/uber/neuropod/blob/master/build/ci_matrix.py#L73-L91)

Code coverage is run on all Buildkite Linux builds

### Lint and Static Analysis

We run the following lint tools in CI:

- [clang-format](https://clang.llvm.org/docs/ClangFormat.html) for C++ formatting
- [Buildifier](https://github.com/bazelbuild/buildtools/tree/master/buildifier) for Bazel BUILD and .bzl file formatting
- [Black](https://black.readthedocs.io/en/stable/) and [Flake8](https://flake8.pycqa.org/en/latest/) for Python lint

To show all lint errors and warnings locally, you can run `./tools/lint.sh`. To attempt to automatically fix any issues that can be automatically fixed, run `./tools/autofix.sh`.

We also run the following static analysis tools in CI:

- [Infer](https://fbinfer.com/) for C++ static analysis
- [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) for C++ lint (Not yet enabled. See [here](https://github.com/uber/neuropod/issues/353))

These tools tend to be fairly slow so we don't currently provide a way to run them locally.

### Future additions

We're also planning on adding the following configurations to the build matrix:

**Configs:**

 - ASAN

## Contributing

See the contributing guide [here](https://github.com/uber/neuropod/blob/master/CONTRIBUTING.md)
