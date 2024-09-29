# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:29:13 2024

@author: Nikita
"""
import torch
import torch.nn as nn
import sys

class Cheby2D(nn.Module):
    """
        Introduces rectangular 2D Chebyshev polynomial.
    """
    def __init__(self, order=4, dtype=torch.complex128, device='cuda:0'):
        super().__init__()
        assert type(order) == int or (type(order) == list and len(order) == 2), \
            "order parameter must be of an int type, or list including 2 ints."
        if type(order) == int:
            self.order = [order, order]
        else:
            self.order = order
        self.dtype = dtype
        self.device = device
        self.vand = None
        param_num = self.order[0] * self.order[1]
        self.weight = torch.nn.Parameter(torch.zeros(param_num, dtype = dtype, device = device), requires_grad = True)
        self.weight.data = 1.e-2 * (torch.rand(param_num, dtype = dtype, device = device) + 1j * torch.rand(param_num, dtype = dtype, device = device) - 1/2 - 1j/2)
        
    def forward(self, input):
        input = torch.abs(input)
        ind1 = torch.arange(self.order[0], device = self.device)
        ind2 = torch.arange(self.order[1], device = self.device)
        
        T0 = torch.cos(ind1[:, None] * torch.arccos(input[0, :1, :]))
        T1 = torch.cos(ind2[:, None] * torch.arccos(input[0, 1:2, :]))
        
        self.vand = (T0[:, None, :] * T1[None, :, :]).reshape(-1, T0.shape[-1]).T.to(self.dtype)

        approx = (self.vand @ self.weight)[None, None, :]
        return approx
    
class Cheby2D_delay(nn.Module):
    def __init__(self, delay, order = 4, complex_coef = True, uniCalc = False, dtype = torch.complex128, device = 'cuda'):
        super().__init__()
        self.delay = Delay(delay)
        self.Cheby = Cheby2D()
        
    def forward(self, input):
        out = self.delay(input)
        return self.Cheby(out)
       
    
    "Rewrite the Delay function"
class DelaySig(nn.Module):
    def __init__(self, M):
        super(Delay, self).__init__()
        self.M = M
        self.op = nn.Sequential(nn.ConstantPad1d(abs(M),0))

    def forward(self, x):
        return self.op(x)[:, :, :x.shape[2]] if self.M > 0 else self.op(x)[:, :, -x.shape[2]:]
    
    
# class Delay(nn.Module):
#     def __init__(self, delays):
#         super(Delay, self).__init__()
#         self.delays = delays

# "Check delay"
# device = 'cuda'
# x = torch.tensor([5, 1, -3, 2, -8, 2, -3, 7, -4, 0]).reshape(1, 1, -1).to(device)
# delay = Delay(-5)
# print(delay(x))
