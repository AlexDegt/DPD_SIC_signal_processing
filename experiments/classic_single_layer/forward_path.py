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
from model import ParallelCheby2D

# Determine experiment name and create its directory
exp_name = "2_param_4_slot_6_cases"
# exp_name = "test"

add_folder = os.path.join("one_dim")
# add_folder = os.path.join("three_dim")
# add_folder = os.path.join("six_dim")
# add_folder = os.path.join("nine_dim")
# add_folder = os.path.join("")
curr_path = os.getcwd()
load_path = os.path.join(curr_path, add_folder, exp_name)
# os.mkdir(save_path)

device = "cuda:5"
# device = "cpu"
seed = 964
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
# data_path = ['../../data/single_band_dynamic/aligned_m0dB_100RB_Fs245p76.mat',]

pa_powers = [0., 0.2, 0.4, 0.6, 0.8, 1.]
# pa_powers = [1.]

# Model initialization
order = [2, 1]
delays = [[j, j, j] for j in range(-15, 16)]
# delays = [[0, 0, 0], [3, 3, 3], [6, 6, 6], [9, 9, 9], [12, 12, 12], [15, 15, 15], [-3, -3, -3], [-6, -6, -6], [-9, -9, -9], [-12, -12, -12], [-15, -15, -15]]
# delays = [[0, 0], [0, 0], [0, 0]]
# Define data type
# dtype = torch.complex64
dtype = torch.complex128
slot_num = 10
# Indices of slots which are chosen to be included in train/test set (must be of a range type).
# Elements of train_slots_ind, test_slots_ind must be higher than 0 and lower, than slot_num
# In full-batch mode train, validation and test dataset are the same.
# In mini-batch mode validation and test dataset are the same.
train_slots_ind, validat_slots_ind, test_slots_ind = range(4), range(4), range(4)
# train_slots_ind, validat_slots_ind, test_slots_ind = range(1), range(1), range(1)
delay_d = 0
# batch_size == None is equal to batch_size = 1.
# block_size == None is equal to block_size = signal length.
# Block size is the same as chunk size 
batch_size = 1
chunk_num = 8
# chunk_size = int(213504/chunk_num)
chunk_size = int(36864 * 6 * 4/chunk_num)
# L2 regularization parameter
alpha = 0.0
# Configuration file
config_train = None
# Input signal is padded with pad_zeros zeros at the beginning and ending of input signal.
# Since each 1d convolution in model CVCNN makes zero-padding with int(kernel_size/2) left and right, then 
# NO additional padding in the input batches is required.
# pad_zeros = 2
pad_zeros = 0
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
#         # print(batch[1].size())
#     print(j + 1)
# sys.exit()

def batch_to_tensors(a):
    x = a[0]
    d = a[1]
    return x, d

def complex_mse_loss(d, y, model):
    error = (d - y)[..., pad_zeros if pad_zeros > 0 else None: -pad_zeros if pad_zeros > 0 else None]
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
        targ_pow += d[..., pad_zeros if pad_zeros > 0 else None: -pad_zeros if pad_zeros > 0 else None].abs().square().sum()
        loss_val += loss(model, batch)
    return 10.0 * torch.log10((loss_val) / (targ_pow)).item()

# def quality_criterion(model, dataset):
#     input_pow, loss_val = 0, 0
#     for batch in dataset:
#         x, _= batch_to_tensors(batch)
#         input_pow += x[..., pad_zeros if pad_zeros > 0 else None: -pad_zeros if pad_zeros > 0 else None].abs().square().sum()
#         loss_val += loss(model, batch)
#     return 10.0 * torch.log10((loss_val) / (input_pow)).item()

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

model = ParallelCheby2D(order, delays, dtype, device)

model.to(device)

weight_names = list(name for name, _ in model.state_dict().items())

print(f"Current model parameters number is {count_parameters(model)}")
# param_names = [name for name, p in model.named_parameters()]
# params = [(name, p.size(), p.dtype) for name, p in model.named_parameters()]
# print(params)

set_weights(model, load_weights(load_path + r'/weights_best_test_' + exp_name))
# set_weights(model, load_weights(load_path + r'/weights_' + exp_name))

model.eval()
with torch.no_grad():
    # train_dataset, validate_dataset, test_dataset
    dataset = validate_dataset
    NMSE = quality_criterion(model, dataset)
    print(NMSE)
    y, d, x = [], [], []
    for j, batch in enumerate(dataset):
        data = batch_to_tensors(batch)

        y.append(model(data[0])[0, 0, :])
        d.append(data[1][0, 0, :])
        x.append(data[0][0, 0, :])

    y_full = torch.cat(y, dim=-1).detach().cpu().numpy()
    d_full = torch.cat(d, dim=-1).detach().cpu().numpy()
    x_full = torch.cat(x, dim=-1).detach().cpu().numpy()[..., pad_zeros if pad_zeros > 0 else None: -pad_zeros if pad_zeros > 0 else None]

    np.save(load_path + r'/y.npy', y_full)
    np.save(load_path + r'/d.npy', d_full)
    np.save(load_path + r'/x.npy', x_full)