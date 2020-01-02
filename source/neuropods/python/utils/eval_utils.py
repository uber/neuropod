#
# Uber, Inc. (c) 2018
#

import logging
import os
import pickle

import numpy as np

from neuropods.utils.env_utils import eval_in_new_process
from neuropods.utils.dtype_utils import maybe_convert_bindings_types

RUN_NATIVE_TESTS = os.getenv("NEUROPOD_RUN_NATIVE_TESTS") is not None
TEST_DATA_FILENAME = "test_data.pkl"

logger = logging.getLogger(__name__)


def check_output_matches_expected(out, expected_out):
    for key, value in expected_out.items():
        if value.dtype.type == np.str_:
            # All strings are equal
            success_condition = (value == out[key]).all()
        else:
            # All the values are close
            success_condition = np.allclose(value, out[key])

        if not success_condition:
            logger.info(
                "Output for key '{}' does not match expected\nExpected:\n{}\nActual\n{}\n".format(
                    key, value, out[key],
                )
            )
            raise ValueError("{} does not match expected value!".format(key))


def print_output_summary(out):
    logger.info("No expected test output specified; printing summary to stdout")
    for key, value in out.items():
        if isinstance(value, np.ndarray):
            logger.info(
                "\t{}: np.array with shape {} and dtype {}".format(
                    key, value.shape, value.dtype
                )
            )
        else:
            raise ValueError(
                "All outputs must be numpy arrays! Output `{}` was of type `{}`".format(
                    key, type(value)
                )
            )


def load_and_test_native(
    neuropod_path, test_input_data, test_expected_out=None, **kwargs
):
    """
    Loads a neuropod using the python bindings and runs inference.
    If expected output is specified, the output of the model is checked against
    the expected values.

    Raises a ValueError if the outputs don't match the expected values
    """
    # Converts unicode to ascii for Python 3
    test_input_data = maybe_convert_bindings_types(test_input_data)

    out = eval_in_new_process(
        neuropod_path, test_input_data, extra_args=["--use-native"], **kwargs
    )

    # Check the output
    if test_expected_out is not None:
        # Throws a ValueError if the output doesn't match the expected value
        check_output_matches_expected(out, test_expected_out)


def load_and_test_neuropod(
    neuropod_path,
    test_input_data,
    test_expected_out=None,
    neuropod_load_args={},
    **kwargs
):
    """
    Loads a neuropod in a new process and verifies that inference runs.
    If expected output is specified, the output of the model is checked against
    the expected values.

    Raises a ValueError if the outputs don't match the expected values
    """
    if RUN_NATIVE_TESTS:
        return load_and_test_native(
            neuropod_path,
            test_input_data,
            test_expected_out,
            neuropod_load_args=neuropod_load_args,
        )

    # Run the evaluation in a new process. This is important to make sure
    # custom ops are being tested correctly
    out = eval_in_new_process(
        neuropod_path, test_input_data, neuropod_load_args=neuropod_load_args
    )

    # Check the output
    if test_expected_out is not None:
        # Throws a ValueError if the output doesn't match the expected value
        check_output_matches_expected(out, test_expected_out)
    else:
        # We don't have any expected output so print a summary
        print_output_summary(out)


def save_test_data(neuropod_path, test_input_data, test_expected_out):
    """
    Saves the neuropods model's test data to a pickle file in the neuropod data directory

    :param neuropod_path: the path of the neuropod model
    :param test_input_data: a dictionary of expected input feature values
    :param test_expected_out: a dictionary of expected output feature values
    :return: None
    """
    test_data = {"test_input": test_input_data, "test_output": test_expected_out}
    with open(os.path.join(neuropod_path, TEST_DATA_FILENAME), "wb") as test_data_file:
        pickle.dump(test_data, test_data_file)


def load_test_data(neuropod_path):
    """
    Loads the neuropods model's test data from the data directory

    :param neuropod_path: the path of the neuropod model
    :return: dict or None
    """
    try:
        with open(
            os.path.join(neuropod_path, TEST_DATA_FILENAME), "rb"
        ) as test_data_file:
            return pickle.load(test_data_file)
    except IOError as err:
        logger.warn("load_test_data IOError {}".format(err))
        return None
