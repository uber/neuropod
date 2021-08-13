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
import os
import unittest

from neuropod.loader import load_neuropod


def get_addition_model_spec(do_fail=False):
    """
    Returns the input/output spec for an addition model along with test data.
    Can also return test data that causes the test to fail.

    :param  do_fail     Return test data that makes the test fail
    """

    return dict(
        input_spec=[
            {"name": "x", "dtype": "float32", "shape": ("batch_size",)},
            {"name": "y", "dtype": "float32", "shape": ("batch_size",)},
        ],
        output_spec=[{"name": "out", "dtype": "float32", "shape": ("batch_size",)},],
        test_input_data={
            "x": np.arange(5, dtype=np.float32),
            "y": np.arange(5, dtype=np.float32),
        },
        test_expected_out={
            "out": np.zeros(5) if do_fail else np.arange(5) + np.arange(5)
        },
    )


def get_string_concat_model_spec(do_fail=False):
    """
    Returns the input/output spec for a string concatenation model along with
    test data. Can also return test data that causes the test to fail.

    :param  do_fail     Return test data that makes the test fail
    """
    if do_fail:
        expected_out = np.array(["a", "b", "c"])
    else:
        expected_out = np.array(["apple sauce", "banana pudding", "carrot cake"])

    return dict(
        input_spec=[
            {"name": "x", "dtype": "string", "shape": ("batch_size",)},
            {"name": "y", "dtype": "string", "shape": ("batch_size",)},
        ],
        output_spec=[{"name": "out", "dtype": "string", "shape": ("batch_size",)},],
        test_input_data={
            "x": np.array(["apple", "banana", "carrot"]),
            "y": np.array(["sauce", "pudding", "cake"]),
        },
        test_expected_out={"out": expected_out,},
    )


def get_mixed_model_spec(do_fail=False):
    """
    Returns the input/output spec for a mixed type model along with test data.
    Can also return test data that causes the test to fail.

    :param  do_fail     Return test data that makes the test fail
    """

    return dict(
        input_spec=[
            {"name": "x", "dtype": "float32", "shape": ("batch_size",)},
            {"name": "y", "dtype": "float32", "shape": ("batch_size",)},
        ],
        output_spec=[
            {"name": "out", "dtype": "float32", "shape": ("batch_size",)},
            {"name": "some", "dtype": "string", "shape": (None,)},
        ],
        test_input_data={
            "x": np.arange(5, dtype=np.float32),
            "y": np.arange(5, dtype=np.float32),
        },
        test_expected_out={
            "out": np.zeros(5) if do_fail else np.arange(5) + np.arange(5),
            "some": np.array(["list", "of", "string"]),
        },
    )


def check_addition_model(neuropod_path):
    """
    Validate that the inputs and outputs of the loaded neuropod match
    the problem spec
    """
    with load_neuropod(neuropod_path) as neuropod:
        target = get_addition_model_spec()

        # Validate that the specs match
        check_specs_match(neuropod.inputs, target["input_spec"])
        check_specs_match(neuropod.outputs, target["output_spec"])
        expected_name = "addition_model"
        if neuropod.name != expected_name:
            raise ValueError(
                "Expected model name '{}'. Got '{}'".format(
                    expected_name, neuropod.name
                )
            )

        if not neuropod.platform:
            raise ValueError("Expected the platform field to be set")


def check_strings_model(neuropod_path):
    """
    Validate that the inputs and outputs of the loaded neuropod match
    the problem spec
    """
    with load_neuropod(neuropod_path, _always_use_native=False) as neuropod:
        target = get_string_concat_model_spec()

        # Validate that the specs match
        check_specs_match(neuropod.inputs, target["input_spec"])
        check_specs_match(neuropod.outputs, target["output_spec"])


def check_specs_match(specs, targets):
    """
    Check that `specs` matches `targets`

    :param  specs       A list of dicts
    :param  targets     A list of dicts
    """
    if len(specs) != len(targets):
        raise ValueError("Length of specs and targets do not match!")

    for spec, target in zip(specs, targets):
        # After loading from JSON, tuples are turned into lists
        target["shape"] = list(target["shape"])

        if spec != target:
            raise ValueError("Spec ({}) not equal to target ({})".format(spec, target))


AVAILABLE_FRAMEWORKS = os.getenv("NEUROPOD_TEST_FRAMEWORKS")

if AVAILABLE_FRAMEWORKS is not None:
    AVAILABLE_FRAMEWORKS = AVAILABLE_FRAMEWORKS.split(",")


def _identity(obj):
    return obj


def requires_frameworks(*frameworks):
    if AVAILABLE_FRAMEWORKS is None:
        return _identity

    for framework in frameworks:
        if framework not in AVAILABLE_FRAMEWORKS:
            return unittest.skip("Skipping test that requires {}".format(framework))

    return _identity
