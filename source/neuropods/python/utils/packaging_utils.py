#
# Uber, Inc. (c) 2019
#

import os
import tempfile
import shutil
import zipfile

from neuropods.backends import config_utils
from neuropods.utils.eval_utils import save_test_data, load_and_test_neuropod

def _zipdir(path, zf):
    for root, dirs, files in os.walk(path):
        for file in files:
            abspath = os.path.join(root, file)
            relpath = os.path.relpath(abspath, path)
            zf.write(abspath, arcname=relpath)

def create_neuropod(
    neuropod_path,
    packager_fn,
    package_as_zip=True,
    custom_ops=[],
    test_input_data=None,
    test_expected_out=None,
    persist_test_data=True,
    **kwargs):
    if package_as_zip:
        package_path = tempfile.mkdtemp()
    else:
        try:
            # Create the neuropod folder
            os.mkdir(neuropod_path)
        except OSError:
            raise ValueError("The specified neuropod path ({}) already exists! Aborting...".format(neuropod_path))

        package_path = neuropod_path

    # Create the structure
    # Store the custom ops (if any)
    neuropod_custom_op_path = os.path.join(package_path, "0", "ops")
    os.makedirs(neuropod_custom_op_path)
    for op in custom_ops:
        shutil.copy(op, neuropod_custom_op_path)

    # Write the neuropod config file
    config_utils.write_neuropod_config(
        neuropod_path=package_path,
        custom_ops=[os.path.basename(op) for op in custom_ops],
        **kwargs
    )

    # Run the packager
    packager_fn(package_path)

    # Persist the test data
    if test_input_data is not None and persist_test_data:
        save_test_data(package_path, test_input_data, test_expected_out)

    # Zip the directory if necessary
    if package_as_zip:
        if os.path.exists(neuropod_path):
            raise ValueError("The specified neuropod path ({}) already exists! Aborting...".format(neuropod_path))

        zf = zipfile.ZipFile(neuropod_path, 'w', zipfile.ZIP_DEFLATED)
        _zipdir(package_path, zf)
        zf.close()

        # Remove our tempdir
        shutil.rmtree(package_path)

    # Test the neuropod
    if test_input_data is not None:
        # Load and run the neuropod to make sure that packaging worked correctly
        # Throws a ValueError if the output doesn't match the expected output (if specified)
        load_and_test_neuropod(
            neuropod_path,
            test_input_data,
            test_expected_out,
            **kwargs
        )
