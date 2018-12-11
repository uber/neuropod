#
# Uber, Inc. (c) 2018
#

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class NeuropodExecutor(object):
    """
    Base class for an Executor
    """

    def infer(self, inputs):
        """
        Run inference using the specifed inputs.

        :param  inputs:     A dict mapping input names to values. This must match the input
                            spec in the neuropod config for the loaded model.
                            Ex: {'x1': [5], 'x2': [6]}

        :returns:   A dict mapping output names to values. This is checked to ensure that it
                    matches the spec in the neuropod config for the loaded model.
        """
        # TODO(vip): validate inputs

        # Run the backend specific inference function
        out = self.forward(inputs)

        # Make sure the key is ascii
        out = {key.encode("ascii"): value for key, value in out.items()}

        # TODO(vip): validate outputs

        return out

    @abc.abstractmethod
    def forward(self, inputs):
        """
        Run inference given a set of inputs. See the docstring for `infer`
        """
        raise NotImplementedError("forward must be implemented by subclasses!")

    def __enter__(self):
        # Needed in order to be used as a contextmanager
        return self

    def __exit__(self, *args):
        # Needed in order to be used as a contextmanager
        pass
