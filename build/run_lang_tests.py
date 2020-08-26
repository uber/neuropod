# Copyright (c) 2020 UATC, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Bazel on Mac does not pass LD_LIBRARY_PATH and DYLD_* to the test runner
# even when using `--test_env`. This may be a bazel bug or be because of system
# integrity protection on Mac.
# Because of this, we're manually running all the bazel tests

import argparse
import re
import subprocess
import sys
import unittest
import os

import xml.etree.ElementTree as ET

class TestBazelTargets(unittest.TestCase):
    pass

# Dynamically generate a unittest given a bazel test target
def make_test(test_target, tags):
    def test(self):
        test_path = test_target.replace('//', './').replace(':', '/')
        runfiles_path = "bazel-bin/" + test_path + ".runfiles/__main__/"

        env = os.environ.copy()
        if "requires_path" in tags:
            # Set PATH for this test
            PATH = env.get("PATH", "")
            PATH += ":" + os.path.join(os.getcwd(), "bazel-bin/neuropod/multiprocess/")
            env["PATH"] = PATH

        if "no_trace_logging" not in tags:
            env["NEUROPOD_LOG_LEVEL"] = "TRACE"

        subprocess.check_call(test_path, cwd=runfiles_path, env=env)
    return test

if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser(description='Run Neuropod tests')

    # --lang flag allows cc, java. Default is empty that means all.
    parser.add_argument('--lang', action='store', default="")
    parser.add_argument('--run-gpu-tests', action='store_true')
    args = parser.parse_args()

    # Get all the bazel test targets that match lang argument
    lang_tests = []
    kind_pattern = 'kind(".*{lang}_test rule", //...)'.format(lang=args.lang)
    lang_tests_xml = subprocess.check_output(['bazel', 'query', kind_pattern, '--output=xml'])

    # Parse the xml
    root = ET.fromstring(lang_tests_xml)
    for child in root:

        # Get tags (if any)
        tags = []
        for prop in child:
            if prop.attrib["name"] == "tags":
                # Got tags
                for tag in prop:
                    tags.append(tag.attrib["value"])

        lang_tests.append((child.attrib["name"], tags))


    # Generate unittests for each bazel test target
    for test_target, tags in lang_tests:
        # Skip GPU tests by default
        if "gpu" in tags and not args.run_gpu_tests:
            continue

        # Name the test and squeeze underscores
        test_name = "test_{}".format(test_target.replace("/", "_").replace(":", "_"))
        test_name = re.sub(r"_+", "_", test_name)

        # Create the test
        print("Creating unittest: {}".format(test_name))
        setattr(TestBazelTargets, test_name, make_test(test_target, tags))

    # Run the tests (and ignore command line args)
    unittest.main(argv=[sys.argv[0]])
