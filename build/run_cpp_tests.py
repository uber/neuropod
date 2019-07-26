#
# Uber, Inc. (c) 2019
#

# Bazel on Mac does not pass LD_LIBRARY_PATH and DYLD_* to the test runner
# even when using `--test_env`. This may be a bazel bug or be because of system
# integrity protection on Mac.
# Because of this, we're manually running all the bazel tests

import re
import subprocess
import unittest

class TestBazelTargets(unittest.TestCase):
    pass

# Dynamically generate a unittest given a bazel test target
def make_test(test_target):
    def test(self):
        test_path = test_target.replace('//', './').replace(':', '/')
        runfiles_path = "bazel-bin/" + test_path + ".runfiles/__main__/"

        subprocess.check_call(test_path, cwd=runfiles_path)
    return test

if __name__ == '__main__':
    # Get all the bazel test targets
    CPP_TESTS = subprocess.check_output(['bazel', 'query', 'kind(".*_test rule", //...)']).splitlines()

    # Generate unittests for each bazel test target
    for test_target in CPP_TESTS:
        # Needed for python 3
        test_target = test_target.decode("utf-8")

        # Name the test and squeeze underscores
        test_name = "test_{}".format(test_target.replace("/", "_").replace(":", "_"))
        test_name = re.sub(r"_+", "_", test_name)

        # Create the test
        print("Creating unittest: {}".format(test_name))
        setattr(TestBazelTargets, test_name, make_test(test_target))

    # Run the tests
    unittest.main()
