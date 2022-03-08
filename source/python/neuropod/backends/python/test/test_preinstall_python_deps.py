# Copyright (c) 2022 The Neuropod Authors
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

import os
import unittest

from neuropod.tests.utils import requires_frameworks
from neuropod.backends.python.utils import preinstall_deps


@requires_frameworks("python")
class TestPreinstallPythonDeps(unittest.TestCase):
    def test_preinstall_python_deps(self):
        NEUROPOD_PYTHON_VERSION = os.getenv("NEUROPOD_PYTHON_VERSION")
        self.assertIsNotNone(
            NEUROPOD_PYTHON_VERSION, "NEUROPOD_PYTHON_VERSION is expected to be set"
        )

        # TODO(vip): This is a bit brittle, but probably isn't worth adding
        # a dependency to the project just to deal with this one case
        major, minor = NEUROPOD_PYTHON_VERSION.split(".")

        # The requirements we're going to be preinstalling
        requirement = "lightgbm==3.3.2"

        # Make sure that the requirement isn't installed before preinstalling it
        # (based on the logic in `pip_utils` in _neuropod_native_bootstrap)
        package_base_dir = os.path.abspath(
            os.path.expanduser(
                "~/.neuropod/pythonpackages/py{}{}/".format(major, minor)
            )
        )

        # Figure out the path that the requirement would be installed to
        # (again, based on the logic in `pip_utils` in _neuropod_native_bootstrap)
        req_path = os.path.abspath(os.path.join(package_base_dir, requirement))

        # TODO(vip): make sure the package isn't installed before this test runs
        # (or delete it if it's already installed)
        # This is somewhat complicated when running tests in parallel so this
        # test is just a sanity check for now

        # Basic test to validate that it installs
        preinstall_deps(NEUROPOD_PYTHON_VERSION, requirements=requirement)

        # Make sure the package was installed
        self.assertTrue(
            os.path.isdir(req_path), "{} should be installed after this test runs"
        )


if __name__ == "__main__":
    unittest.main()
