import os
import unittest
from testpath.tempdir import TemporaryDirectory

from neuropods.utils.eval_utils import save_test_data, load_test_data


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
