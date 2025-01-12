import sys

sys.path.append('../../')
import os 

import torch
import random
import numpy as np
from oracle import count_parameters
from trainer import train
from utils import dynamic_dataset_prepare
from scipy.io import loadmat
from model import CVCNN

seed = 964
# Determine experiment name and create its directory
exp_name = "sgd_test"

add_folder = os.path.join("")
curr_path = os.getcwd()
load_path = os.path.join(curr_path, add_folder, exp_name)
# os.mkdir(save_path)

device = "cuda:6"
# device = "cpu"
# seed = 964
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# torch.use_deterministic_algorithms(True)
if device != "cpu":
    torch.backends.cudnn.deterministic = True

# Load PA input and output data. Data for different cases is concatenated together
data_path = ['../../data/single_band_dynamic/aligned_m15dB_100RB_Fs245p76.mat',
             '../../data/single_band_dynamic/aligned_m12dB_100RB_Fs245p76.mat',
             '../../data/single_band_dynamic/aligned_m9dB_100RB_Fs245p76.mat',
             '../../data/single_band_dynamic/aligned_m6dB_100RB_Fs245p76.mat',
             '../../data/single_band_dynamic/aligned_m3dB_100RB_Fs245p76.mat',
             '../../data/single_band_dynamic/aligned_m0dB_100RB_Fs245p76.mat',]
# data_path = ['../../data/single_band_dynamic/aligned_m15dB_100RB_Fs245p76.mat']
# data_path = ['../../data/single_band_dynamic/aligned_m0dB_100RB_Fs245p76.mat',
#              '../../data/single_band_dynamic/aligned_m15dB_100RB_Fs245p76.mat']

# pa_powers = [1., 0.]
# pa_powers = [0.]
pa_powers = [0., 0.2, 0.4, 0.6, 0.8, 1.]

# Define data type
# dtype = torch.complex64
dtype = torch.complex128

# Number of output channels of each convolutional layer.
# out_channels = [1, 1]
out_channels = [3, 3, 3, 2]
# out_channels = [8, 8, 8, 8, 12, 2]
# Kernel size for each layer. Kernel sizes must be odd integer numbers.
# Otherwise input sequence length will be reduced by 1 (for case of mode == 'same' in nn.Conv1d).
# kernel_size = [3, 3]
kernel_size = [3, 3, 3, 3]
# kernel_size = [5, 5, 5, 5, 5, 5]
# Activation functions are listed in /model/layers/activation.py
# Don`t forget that model output must be holomorphic w.r.t. model parameters
# activate = ['sigmoid', 'sigmoid']
activate = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
# activate = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh']
# activate = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'pass_act', 'pass_act']
# activate = ['ctanh', 'ctanh', 'ctanh', 'ctanh']
p_drop = list(np.zeros_like(out_channels))
delays = [[0]]
slot_num = 10
# Indices of slots which are chosen to be included in train/test set (must be of a range type).
# Elements of train_slots_ind, test_slots_ind must be higher than 0 and lower, than slot_num
# In full-batch mode train, validation and test dataset are the same.
# In mini-batch mode validation and test dataset are the same.
train_slots_ind, validat_slots_ind, test_slots_ind = range(1), range(1), range(1)
delay_d = 0
# batch_size == None is equal to batch_size = 1.
# block_size == None is equal to block_size = signal length.
# Block size is the same as chunk size 
batch_size = 1
chunk_num = 1
# chunk_size = int(213504/chunk_num)
chunk_size = int(36864 * 6/chunk_num)
# L2 regularization parameter
alpha = 0.0
# Configuration file
config_train =  None
# Input signal is padded with pad_zeros zeros at the beginning and ending of input signal.
# Since each 1d convolution in model CVCNN makes zero-padding with int(kernel_size/2) left and right, then 
# NO additional padding in the input batches is required.
# pad_zeros = 2
pad_zeros = sum([kernel // 2 for kernel in kernel_size])
dataset = dynamic_dataset_prepare(data_path, pa_powers, dtype, device, slot_num=slot_num, delay_d=delay_d,
                          train_slots_ind=train_slots_ind, test_slots_ind=test_slots_ind, validat_slots_ind=validat_slots_ind,
                          pad_zeros=pad_zeros, batch_size=batch_size, block_size=chunk_size)

train_dataset, validate_dataset, test_dataset = dataset

# Show sizes of batches in train dataset, size of validation and test dataset
# for i in range(len(dataset)):
#     for j, batch in enumerate(dataset[i]):
#         # if j == 0:
#         # Input batch size
#         print(batch[0].size())
#         # Target batch size
#         print(batch[1].size())
#     print(j + 1)
# sys.exit()

def batch_to_tensors(a):
    x = a[0]
    d = a[1]
    return x, d

def complex_mse_loss(d, y, model):
    error = (d - y)[..., pad_zeros: -pad_zeros]
    # error = (d - y)
    return error.abs().square().sum() #+ alpha * sum(torch.norm(p)**2 for p in model.parameters())

def loss(model, signal_batch):
    x, y = batch_to_tensors(signal_batch)
    return complex_mse_loss(model(x), y, model)
# This function is used only for telecom task.
# Calculates NMSE on base of accumulated on every batch loss function
@torch.no_grad()
# To avoid conflicts for classification task you can write:
# def quality_criterion(loss_val):
#     return loss_val
def quality_criterion(model, dataset):
    targ_pow, loss_val = 0, 0
    for batch in dataset:
        _, d= batch_to_tensors(batch)
        targ_pow += d[..., pad_zeros: -pad_zeros].abs().square().sum()
        loss_val += loss(model, batch)
    return 10.0 * torch.log10((loss_val) / (targ_pow)).item()

def load_weights(path_name, device=device):
    return torch.load(path_name, map_location=torch.device(device))

def set_weights(model, weights):
    model.load_state_dict(weights)

def get_nested_attr(module, names):
    for i in range(len(names)):
        module = getattr(module, names[i], None)
        if module is None:
            return
    return module

# CVCNN - Complex-Valued Convolutional NN.
# Takes pure signal both channels x_{A, n}, x_{B, n} as an input and 
# creates input features: x_{A, n}, x_{B, n}, |x_{A, n}|, |x_{B, n}|. Thus there're 4 input channels.
# Output channel numbers are rehulated by the list out_channels.
# Last layer output channels number equal 1, which corresponds to pre-distorted signal.
model = CVCNN(device=device, delays=delays, out_channels=out_channels, kernel_size=kernel_size, features=['same', 'abs'], 
              activate=activate, batch_norm_mode='nothing', p_drop=p_drop, bias=True, dtype=dtype)

model.to(device)

weight_names = list(name for name, _ in model.state_dict().items())

print(f"Current model parameters number is {count_parameters(model)}")

set_weights(model, load_weights(load_path + r'/weights_best_test_' + exp_name))
# set_weights(model, load_weights(load_path + r'/weights_' + exp_name))

model.eval()
with torch.no_grad():
    # train_dataset, validate_dataset, test_dataset
    dataset = validate_dataset
    NMSE = quality_criterion(model, dataset)
    print(NMSE)
    for j, batch in enumerate(dataset):
        data = batch_to_tensors(batch)
    y = model(data[0])

y = y[0, 0, :].detach().cpu().numpy()
d = data[1][0, 0, :].detach().cpu().numpy()
np.save(load_path + r'/y.npy', y)
np.save(load_path + r'/d.npy', d)