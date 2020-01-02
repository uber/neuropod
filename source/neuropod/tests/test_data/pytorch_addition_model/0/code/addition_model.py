import torch.nn as nn


class AdditionModel(nn.Module):
    def forward(self, x, y):
        return {"out": x + y}


def get_model(_):
    return AdditionModel()
