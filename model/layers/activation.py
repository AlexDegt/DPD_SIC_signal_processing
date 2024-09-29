import torch
import torch.nn as nn
from torch.nn import ReLU, LeakyReLU, PReLU

class pass_act(nn.Module):
    def __init__(self):
        super(pass_act, self).__init__()
    def forward(self, x):
        return x

class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()
    def forward(self, x):
        return torch.tanh(x) 

class CTanh(nn.Module):
    def __init__(self):
        super(CTanh, self).__init__()
    def forward(self, x):
        # Attention! In case of layer_number > 1 torch.tanh(x)
        # torch.autograd.functional.jacobian doesn`t propagate gradient through exp and tanh
        # out = torch.tanh(x) 
        exp_2x = torch.exp(2 * x.real)
        exp_2jy = torch.exp(-2 * 1j * x.imag)
        out = (exp_2x - exp_2jy)/(exp_2x + exp_2jy)
        return out

class lim_lin(nn.Module):
    ''' Limited-linear activation '''
    def __init__(self):
        super(lim_lin, self).__init__()
    def forward(self, x):
        x[x < -1] = -1
        x[x > 1] = 1
        return x

class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
    def forward(self, x):
        return 2*torch.sigmoid(x) - 1

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()
        self.relu = ReLU()
    def forward(self, x):
        return self.relu(x.real) + 1j*self.relu(x.imag)

class CPReLU(nn.Module):
    def __init__(self):
        super(CPReLU, self).__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1, dtype=torch.complex128), requires_grad=True)
    def forward(self, x):
        x[torch.abs(x+self.bias) < 0] = 0+0j
        return torch.abs(x+self.bias)*torch.exp(1j*torch.angle(x))

def configure_activates(activate_str: str, channel_num: int=8, dtype=torch.complex128, device='cuda'):
    if activate_str == 'tanh':
        activate_func = Tanh()
    if activate_str == 'ctanh':
        activate_func = CTanh()
    if activate_str == 'relu':
        activate_func = ReLU()
    if activate_str == 'leaky_relu':
        activate_func = LeakyReLU(negative_slope=1e-1)
    if activate_str == 'CReLU':
        activate_func = CReLU()
    if activate_str == 'CPReLU':
        activate_func = CPReLU()
    if activate_str == 'PReLU':
        activate_func = PReLU(num_parameters=2)
    if activate_str == 'lim_lin':
        activate_func = lim_lin()
    if activate_str == 'sigmoid':
        activate_func = Sigmoid()
    if activate_str == 'pass_act':
        activate_func = pass_act()
    return activate_func