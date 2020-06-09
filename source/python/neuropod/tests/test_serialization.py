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

import numpy as np
from neuropod import neuropod_native
import six
import unittest


class TestSerialization(unittest.TestCase):
    def test_different_types_and_shapes(self):
        """
        Exhaustive noop serialization-->deserialization test over all supported datatypes and several distrinct
        shapes
        """
        _TESTED_DTYPES = [
            np.float32,
            np.float64,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.string_,
        ]
        _TESTED_SHAPES = [(0,), (1,), (3,), (2, 3), (2, 3, 4)]

        # Used for testing NeuropodValueMap serialization
        counter = 0
        expected_valuemap = {}

        for dtype in _TESTED_DTYPES:
            for shape in _TESTED_SHAPES:
                expected = np.random.random(size=shape).astype(dtype=dtype)
                buffer = neuropod_native.serialize(expected)
                actual = neuropod_native.deserialize(buffer)

                # Unicode vs ascii for python 3
                # TODO(vip): Add unicode support to the native bindings and fix this
                if actual.dtype.type == np.unicode_:
                    actual = actual.astype(np.string_)

                np.testing.assert_array_equal(expected, actual)

                # Add it to our dict to test NeuropodValueMap serialization
                expected_valuemap["item{}".format(counter)] = expected
                counter += 1

        # Test serializing and deserializing a dict of numpy arrays
        buffer = neuropod_native.serialize(expected_valuemap)
        actual_valuemap = neuropod_native.deserialize_dict(buffer)

        # Make sure they're the same size
        self.assertEqual(len(expected_valuemap), len(actual_valuemap))

        # Compare elements
        for key, expected in expected_valuemap.items():
            actual = actual_valuemap[key]

            # Unicode vs ascii for python 3
            # TODO(vip): Add unicode support to the native bindings and fix this
            if actual.dtype.type == np.unicode_:
                actual = actual.astype(np.string_)

            np.testing.assert_array_equal(expected, actual)

    def test_invalid_stream_deserialization(self):
        with self.assertRaises(RuntimeError if six.PY2 else TypeError):
            neuropod_native.deserialize("bogus")


if __name__ == "__main__":
    unittest.main()
