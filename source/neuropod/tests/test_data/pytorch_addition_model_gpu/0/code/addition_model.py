import torch
import torch.nn as nn


class AdditionModel(nn.Module):
    def forward(self, x, y):
        x = torch.from_numpy(x).cuda(0)
        y = torch.from_numpy(y).cuda(0)
        return {"out": (x + y).cpu().numpy()}


def get_model(_):
    return AdditionModel()
