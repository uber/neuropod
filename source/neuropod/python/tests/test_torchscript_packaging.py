#
# Uber, Inc. (c) 2018
#

import os
import torch
import unittest
from subprocess import CalledProcessError
from testpath.tempdir import TemporaryDirectory
from torch import Tensor
from typing import Dict

from neuropod.packagers import create_torchscript_neuropod
from neuropod.tests.utils import (
    get_addition_model_spec,
    get_mixed_model_spec,
    check_addition_model,
)


class AdditionModel(torch.jit.ScriptModule):
    """
    A simple addition model
    """

    @torch.jit.script_method
    def forward(self, x, y):
        return {"out": x + y}


class AdditionModelDictInput(torch.jit.ScriptModule):
    """
    A simple addition model.
    If there are a large number of inputs, it may be more convenient to take the
    input as a dict rather than as individual parameters.
    """

    @torch.jit.script_method
    def forward(self, data):
        # type: (Dict[str, Tensor])
        return {"out": data["x"] + data["y"]}


class AdditionModelTensorOutput(torch.jit.ScriptModule):
    """
    A simple addition model
    """

    @torch.jit.script_method
    def forward(self, x, y):
        return x + y


class MixedReturnTypesModel(torch.jit.ScriptModule):
    """
    Torchscript dictionaries must be of a single type (e.g. Dict[str, Tensor])
    This means we can't mix tensors with "string tensors" (i.e. list of strings)

    This is solved by returning multiple dicts
    """

    @torch.jit.script_method
    def forward(self, x, y):
        # A dict containing our tensor outputs
        tensor_output = {"out": x + y}

        # A dict containing our "string tensor" outputs
        string_output = {"some": ["list", "of", "string"]}

        return tensor_output, string_output


class MixedReturnTypesModelDuplicateItem(torch.jit.ScriptModule):
    """
    NOTE: This model is intended to cause a failure. See below
    """

    @torch.jit.script_method
    def forward(self, x, y):
        # A dict containing our tensor outputs
        tensor_output = {"out": x + y}

        # A dict containing our "string tensor" outputs
        string_output = {"some": ["list", "of", "string"]}

        # This should cause an error because it's overwriting `out`
        # from the above dict
        tensor_output_2 = {"out": x + y}

        return tensor_output, string_output, tensor_output_2


class TestTorchScriptPackaging(unittest.TestCase):
    def package_simple_addition_model(self, do_fail=False):
        for model in [AdditionModel, AdditionModelDictInput, AdditionModelTensorOutput]:
            with TemporaryDirectory() as test_dir:
                neuropod_path = os.path.join(test_dir, "test_neuropod")

                # `create_torchscript_neuropod` runs inference with the test data immediately
                # after creating the neuropod. Raises a ValueError if the model output
                # does not match the expected output.
                create_torchscript_neuropod(
                    neuropod_path=neuropod_path,
                    model_name="addition_model",
                    module=model(),
                    # Get the input/output spec along with test data
                    **get_addition_model_spec(do_fail=do_fail)
                )

                # Run some additional checks
                check_addition_model(neuropod_path)

    def test_simple_addition_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        self.package_simple_addition_model()

    def test_simple_addition_model_failure(self):
        # Tests a case where the output does not match the expected output
        with self.assertRaises(ValueError):
            self.package_simple_addition_model(do_fail=True)

    def test_mixed_types_model(self):
        # Tests a model that returns both tensors and "string tensors"
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

            # `create_torchscript_neuropod` runs inference with the test data immediately
            # after creating the neuropod. Raises a ValueError if the model output
            # does not match the expected output.
            create_torchscript_neuropod(
                neuropod_path=neuropod_path,
                model_name="mixed_types_model",
                module=MixedReturnTypesModel(),
                # Get the input/output spec along with test data
                **get_mixed_model_spec()
            )

    def test_mixed_types_model_failure(self):
        # Tests a model that returns both tensors and "string tensors"
        with self.assertRaises(ValueError):
            with TemporaryDirectory() as test_dir:
                neuropod_path = os.path.join(test_dir, "test_neuropod")

                # `create_torchscript_neuropod` runs inference with the test data immediately
                # after creating the neuropod. Raises a ValueError if the model output
                # does not match the expected output.
                create_torchscript_neuropod(
                    neuropod_path=neuropod_path,
                    model_name="mixed_types_model",
                    module=MixedReturnTypesModel(),
                    # Get the input/output spec along with test data
                    **get_mixed_model_spec(do_fail=True)
                )

    def test_mixed_types_model_failure_duplicate_item(self):
        # Tests a model that returns duplicate items across multiple dictionaries
        # This is either a CalledProcessError or a RuntimeError depending on whether
        # we're using the native bindings or not
        with self.assertRaises((CalledProcessError, RuntimeError)):
            with TemporaryDirectory() as test_dir:
                neuropod_path = os.path.join(test_dir, "test_neuropod")

                # `create_torchscript_neuropod` runs inference with the test data immediately
                # after creating the neuropod. Raises a ValueError if the model output
                # does not match the expected output.
                create_torchscript_neuropod(
                    neuropod_path=neuropod_path,
                    model_name="mixed_types_model",
                    module=MixedReturnTypesModelDuplicateItem(),
                    # Get the input/output spec along with test data
                    **get_mixed_model_spec()
                )


if __name__ == "__main__":
    unittest.main()
