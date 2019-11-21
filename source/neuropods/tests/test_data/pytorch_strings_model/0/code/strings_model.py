import numpy as np
import torch
import torch.nn as nn


class StringsModel(nn.Module):
    def forward(self, x, y):
        return {"out": np.array([f + " " + s for f, s in zip(x, y)])}


def get_model(_):
    return StringsModel()
