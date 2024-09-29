import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import numpy as np
import sys

class Delay(nn.Module):
    def __init__(self, delays, dtype=torch.complex128, device=None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.delays = delays
        # self.delays = list(chain(*delays))
        self.branch_num = len(delays)
        self.delays_num = len(delays[0])

    def forward(self, x):
        assert x.shape[1] == self.delays_num, "Number of channels of input signal must equal the number of delays in each branch of model."
        output = torch.zeros(x.shape[0], self.branch_num, self.delays_num, x.shape[2], dtype=self.dtype, device=self.device)
        for j_branch in range(self.branch_num):
            for j_delay, delay in enumerate(self.delays[j_branch]):
                output[:, j_branch, j_delay, :] = torch.roll(x[:, j_delay, :], shifts=-1*delay, dims=-1)
        return output