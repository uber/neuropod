import numpy as np
import torch
import torch.nn as nn


class StringsModel(nn.Module):
    def forward(self, x, y):
        # Python 3 uses unicode for strings. Neuropod doesn't automatically convert
        # string tensors to unicode because it's possible that these tensors contain
        # aribtrary byte sequences instead of valid unicode.

        # Since we know x and y contain valid strings, we can convert to np.str_ here.
        # This decodes the data as unicode in python3 and doesn't have an effect in
        # python2.
        x = x.astype(np.str_)
        y = y.astype(np.str_)

        return {
            "out": np.array([f + " " + s for f, s in zip(x, y)])
        }


def get_model(_):
    return StringsModel()
