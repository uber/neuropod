#
# Uber, Inc. (c) 2019
#

import os
import tempfile
import shutil

from neuropods.backends import config_utils
from neuropods.utils.eval_utils import save_test_data, load_and_test_neuropod

def create_neuropod(
    neuropod_path,
    packager_fn,
    custom_ops=[],
    test_input_data=None,
    test_expected_out=None,
    persist_test_data=True,
    **kwargs):
    try:
        # Create the neuropod folder
        os.mkdir(neuropod_path)
    except OSError:
        raise ValueError("The specified neuropod path ({}) already exists! Aborting...".format(neuropod_path))

    # Create the structure
    # Store the custom ops (if any)
    neuropod_custom_op_path = os.path.join(neuropod_path, "0", "ops")
    os.makedirs(neuropod_custom_op_path)
    for op in custom_ops:
        shutil.copy(op, neuropod_custom_op_path)

    # Write the neuropod config file
    config_utils.write_neuropod_config(
        neuropod_path=neuropod_path,
        custom_ops=[os.path.basename(op) for op in custom_ops],
        **kwargs
    )

    # Run the packager
    packager_fn(neuropod_path)

    # Test the neuropod
    if test_input_data is not None:
        if persist_test_data:
            save_test_data(neuropod_path, test_input_data, test_expected_out)
        # Load and run the neuropod to make sure that packaging worked correctly
        # Throws a ValueError if the output doesn't match the expected output (if specified)
        load_and_test_neuropod(
            neuropod_path,
            test_input_data,
            test_expected_out,
            **kwargs
        )
