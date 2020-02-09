#
# Uber, Inc. (c) 2019
#

# This script lets us define a CI build matrix in one place
import itertools

# Template for .travis.yml
TRAVIS_TEMPLATE = """
#
# DO NOT MANUALLY EDIT THIS FILE
# AUTOGENERATED BY build/ci_matrix.py
#

# Run on 16.04 for linux builds
dist: xenial
language: minimal

# Define a build matrix
matrix:
  include:
    # A single linux build to make sure that the build scripts work correctly outside of docker
    - os: linux
{}

script:
  # Travis changes whether the shell is a TTY or not depending on if there are secret env variables set.
  # Travis also does not set these variables for PRs from outside the repo (e.g. from forks) for security
  # reasons (https://docs.travis-ci.com/user/pull-requests/#pull-requests-and-security-restrictions).
  # This leads to different behavior depending on where the PR came from.
  # To get around this, we always run in a non-interactive shell.
  - set -o pipefail
  - ./build/ci/travis_build.sh < /dev/null | cat
"""

# Template for docker-compose.test.yml
DOCKER_COMPOSE_TEST_TEMPLATE = """
#
# DO NOT MANUALLY EDIT THIS FILE
# AUTOGENERATED BY build/ci_matrix.py
#

version: '2.3'
services:
  test-base:
    build:
      context: .
      dockerfile: build/neuropod.dockerfile
      target: neuropod-base
    privileged: true

  test-gpu:
    extends: test-base
    build:
      args:
        NEUROPOD_IS_GPU: "true"
    runtime: nvidia

{}
"""

# Template for .buildkite/pipeline.yml
BUILDKITE_YML_TEMPLATE = """
#
# DO NOT MANUALLY EDIT THIS FILE
# AUTOGENERATED BY build/ci_matrix.py
#

steps:
{}
"""

# The platforms we're testing on
PLATFORMS = [
    "linux_cpu",
    "linux_gpu",
    "macos_cpu",
]

# Versions of python to use
PY_VERSIONS = [
    "2.7",
    "3",
]

# Versions of frameworks to test with
FRAMEWORK_VERSIONS = [
    {"cuda": "9.0", "tensorflow": "1.12.0", "torch": "1.1.0"},
    {"cuda": "10.0", "tensorflow": "1.13.1", "torch": "1.2.0"},
    {"cuda": "10.0", "tensorflow": "1.14.0", "torch": "1.3.0"},
    {"cuda": "10.0", "tensorflow": "1.15.0", "torch": "1.4.0"},
]

travis_matrix = []
docker_compose_matrix = []
buildkite_yml_matrix = []
added_lint = False
for platform, py_version, framework_version in itertools.product(PLATFORMS, PY_VERSIONS, FRAMEWORK_VERSIONS):
    # Get versions of all the dependencies
    tf_version = framework_version["tensorflow"]
    torch_version = framework_version["torch"]
    py_binary = "python" if py_version == "2.7" else "python3"

    # Generate the appropriate configuration
    if "macos" in platform:
        # This is a Travis CI build
        travis_matrix.extend([
        "    - os: osx\n",
        "      env: NEUROPOD_TENSORFLOW_VERSION={} NEUROPOD_TORCH_VERSION={} NEUROPOD_PYTHON_BINARY={}\n".format(tf_version, torch_version, py_binary),
        "\n",
        ])

    elif "linux" in platform:
        is_gpu = "gpu" in platform
        variant_name = "test-{}-variant-tf-{}-torch-{}-py{}".format("gpu" if is_gpu else "cpu", tf_version, torch_version, py_version).replace(".", "_")

        docker_compose_matrix.extend([
        "  {}:\n".format(variant_name),
        "    extends: test-{}\n".format("gpu" if is_gpu else "base"),
        "    build:\n",
        "      args:\n",
        "        NEUROPOD_CUDA_VERSION: {}\n".format(framework_version["cuda"]) if is_gpu else "",
        "        NEUROPOD_TENSORFLOW_VERSION: {}\n".format(tf_version),
        "        NEUROPOD_TORCH_VERSION: {}\n".format(torch_version),
        "        NEUROPOD_PYTHON_BINARY: {}\n".format(py_binary),
        "\n",
        ])

        plugin_config = [
        "    plugins:\n",
        "      - docker-compose#v3.1.0:\n",
        "          build: {}\n".format(variant_name),
        "          config: docker-compose.test.yml\n",
        "          image-repository: 027047743804.dkr.ecr.us-east-2.amazonaws.com/uber\n",
        "          cache-from: {}:027047743804.dkr.ecr.us-east-2.amazonaws.com/uber:{}\n".format(variant_name, variant_name),
        "          push-retries: 5\n",
        "      - docker-compose#v3.1.0:\n",
        "          push: {}:027047743804.dkr.ecr.us-east-2.amazonaws.com/uber:{}\n".format(variant_name, variant_name),
        "          config: docker-compose.test.yml\n",
        "      - docker-compose#v3.1.0:\n",
        "          run: {}\n".format(variant_name),
        "          config: docker-compose.test.yml\n",
        "          env:\n",
        "            - NEUROPOD_CACHE_ACCESS_KEY\n",
        "            - NEUROPOD_CACHE_ACCESS_SECRET\n",
        "            - BUILDKITE\n",
        "            - BUILDKITE_BRANCH\n",
        "            - BUILDKITE_BUILD_NUMBER\n",
        "            - BUILDKITE_BUILD_URL\n",
        "            - BUILDKITE_COMMIT\n",
        "            - BUILDKITE_JOB_ID\n",
        "            - BUILDKITE_PROJECT_SLUG\n",
        "            - BUILDKITE_PULL_REQUEST\n",
        "            - BUILDKITE_TAG\n",
        "            - CI\n",
        "            - CODECOV_TOKEN\n",
        "            - GH_STATUS_TOKEN\n",
        "            - GH_UPLOAD_TOKEN\n",
        "    retry:\n",
        "      automatic: true\n",
        "\n",
        ]

        buildkite_yml_matrix.extend([
        "  - label: \":docker: {} Tests ({})\"\n".format("GPU" if is_gpu else "CPU", variant_name),
        "    agents:\n",
        "      queue: private-{}\n".format("gpu" if is_gpu else "default"),
        "    command: build/ci/{}.sh\n".format("buildkite_build_gpu" if is_gpu else "buildkite_build"),
        ] + plugin_config)

        if not is_gpu and not added_lint:
            # Add a lint job to our build matrix
            added_lint = True

            buildkite_yml_matrix.extend([
            "  - label: \":docker: Lint + Docs\"\n".format(variant_name),
            "    agents:\n",
            "      queue: private-default\n",
            "    command: build/ci/buildkite_lint.sh\n",
            ] + plugin_config)


# Use the templates to create the complete config files
TRAVIS_YML = TRAVIS_TEMPLATE.format("".join(travis_matrix))
DOCKER_COMPOSE_TEST = DOCKER_COMPOSE_TEST_TEMPLATE.format("".join(docker_compose_matrix))
BUILDKITE_YML = BUILDKITE_YML_TEMPLATE.format("".join(buildkite_yml_matrix))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-config', help=('Actually write all the CI config files (e.g. pipeline.yml, .travis.yml, docker-compose.test.yml).'
                                              'Otherwise, just verify that the files match the build matrix defined here'
                                              'Default False.'), default=False, action='store_true')
    args = parser.parse_args()

    files = {
        './docker-compose.test.yml': DOCKER_COMPOSE_TEST,
        './.travis.yml': TRAVIS_YML,
        './.buildkite/pipeline.yml': BUILDKITE_YML,
    }

    if args.write_config:
        for path, content in files.items():
            with open(path, 'w') as f:
                f.write(content)
    else:
        # Just verify that everything matches
        for path, target_content in files.items():
            with open(path, 'r') as f:
                content = f.read()
                if content != target_content:
                    raise ValueError("{} does not match current build matrix! Please run `./build/ci_matrix.py --write-config` to fix.".format(path))
