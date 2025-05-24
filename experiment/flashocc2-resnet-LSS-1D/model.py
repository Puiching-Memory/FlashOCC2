import sys
import os

sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models import block


class model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> tuple:

        return (
        )


if __name__ == "__main__":
    pass
