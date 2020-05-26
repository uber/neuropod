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

import os
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropod.utils.eval_utils import save_test_data, load_test_data


class TestEvalUtil(unittest.TestCase):
    def test_save_load_test_data(self):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")
            os.mkdir(neuropod_path)

            test_input = {"x": 3, "y": 4}
            test_expected_output = {"out": 7}

            save_test_data(neuropod_path, test_input, test_expected_output)
            test_data = load_test_data(neuropod_path)
            self.assertEquals(test_input, test_data["test_input"])
            self.assertEquals(test_expected_output, test_data["test_output"])
