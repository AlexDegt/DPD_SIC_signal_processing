import torch
from typing import Tuple, Union, Iterable, List
import torch.nn.functional as F
import scipy.signal as signal
from scipy.io import loadmat
import sys
import numpy as np
from scipy.io import loadmat

OptionalTensor = Union[torch.Tensor, None]
OptionalStr = Union[str, None]
OptionalInt = Union[int, None]
DatasetType = Union[Tuple[Iterable, ...], Tuple[Tuple[Iterable, ...], ...]]
ListOfStr = List[str]
ListOfFloat = List[float]

class ResampleDataset(torch.utils.data.Dataset):
    """
    The dataset class that extracts batches in a tuple type, where the first
    tuple element contains input batch part, the second tuple element contains 
    target batch part.
    """
    def __init__(self, data: Tuple[torch.Tensor], batch_size: OptionalInt = None, downsample_ratio: OptionalInt = None):
        super(ResampleDataset, self).__init__()
        if downsample_ratio is None:
            downsample_ratio = 1
        if batch_size is None:
            batch_size = 1
            self.batch_num = 1
        else:
            self.batch_num = int(np.ceil(data[0].shape[0]/batch_size))
        self.data = tuple((data[0], data[1]))
        self.batch_size = int(batch_size)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        if index < self.batch_num -  1:
            return tuple((self.data[0][index*self.batch_size:(index+1)*self.batch_size, ...], 
                            self.data[1][index*self.batch_size:(index+1)*self.batch_size, ...]))
        if index == self.batch_num -  1:
            return tuple((self.data[0][index*self.batch_size:, ...], 
                            self.data[1][index*self.batch_size:, ...]))
    def __len__(self) -> int:
        return self.batch_num

def dynamic_dataset_prepare(data_path: ListOfStr, pa_powers: ListOfFloat, dtype: torch.dtype = torch.complex128, device: str = 'cuda', batch_size: OptionalInt = None, 
                    block_size: OptionalInt = None, slot_num: OptionalInt = None, pad_zeros: OptionalInt = None, 
                    delay_d: OptionalInt = None, train_slots_ind: range = range(1), validat_slots_ind: range = range(1),
                    test_slots_ind: range = range(1)) -> DatasetType:
    """
    The method extracts input and target data for the mat file, normalizes and resamples if necessary.
    Then it divides input and target tensors into the batches and loads them into the dataloader.

    Args:
        data_path (list of str): List of pathes to mat-files with dynamic data. Each mat-file contains:
            input (numpy.ndarray): 1d array with shape (1, arr.shape), which contains input data samples.
            target (numpy.ndarray): 1d array with shape (1, arr.shape), which contains target data samples.
            noise_floor (numpy.ndarray): 1d array with shape (1, arr.shape), which contains noise floor samples.
        pa_powers (list of float): List PA output powers, which correspond to data within data_path mat-files.
        dtype (torch.dtype): The type of tensor to convert content of the mat-file to. Defaults is torch.complex128.
        device (str): The device to load dataset to. Defaults is 'cpu'.
        slot_num (int, optional): The number of slots to divide the whole dataset into.
        pad_zeros (int, optional): The number of zeros to add to the beginning and to the end of the signal.
        delay_d (int, optional): The value of desired and noise floor signals shift. delay_d > 0 corresponds to
            samples shift left, delay_d < 0 corresponds to samples shift right. If delay_d equal zero or None, 
            then there is no shift. Defaults is "None".
        train_slots_ind (range): Indices of the slots which are chosen for training dataset. A range with step 1. Defaults is range(1).
        validat_slots_ind (range): Used only for hold-out cross-validation. Indices of the slots which are chosen for validation dataset. 
            A range with step 1. Defaults is range(1).
        test_slots_ind (range): Indices of the slots which are chosen for training dataset. A range with step 1. Defaults is range(1).
            
    Returns:
        Tuple of iterables.
    """
    
    if pad_zeros is None:
        pad_zeros = 0

    if batch_size is None:
        batch_size = 1

    assert len(data_path) == len(pa_powers), "Number of dynamic cases in data_path must equal number of corresponding PA output powers."
    dynam_case_num = len(data_path)

    input, target = [], []
    for path in data_path:
        mat = loadmat(path)
        input_tensor = torch.tensor(mat['TX'][0, :], dtype=dtype).view(1, 1, -1).to(device)
        target_tensor = torch.tensor(mat['PAout'][0, :] - mat['TX'][0, :], dtype=dtype).view(1, 1, -1).to(device)
        input.append(input_tensor)
        target.append(target_tensor)

    pa_list = [pa_pow * torch.ones(1, 1, input[0].numel()) for pa_pow in pa_powers]
    pa_powers = torch.cat(pa_list, dim=1).to(device).to(dtype)


    input = torch.cat(input, dim=1)
    target = torch.cat(target, dim=1)

    input = torch.cat([input[:, None, ...], pa_powers[:, None, ...]], dim=1)

    if delay_d is not None and delay_d != 0:
        target = torch.roll(target, -delay_d, dims=-1)

    input = input / 1
    target = target / 1

    assert (np.array(train_slots_ind) < slot_num).all() and (np.array(train_slots_ind) >= 0).all(), \
        "All train slots indices (argument train_slots_ind) must be positive and lower, than number of slots (argument slot_num)."
    assert (np.array(validat_slots_ind) < slot_num).all() and (np.array(validat_slots_ind) >= 0).all(), \
        "All validation slots indices (argument validat_slots_ind) must be postive and lower, than number of slots (argument slot_num)."
    assert (np.array(test_slots_ind) < slot_num).all() and (np.array(test_slots_ind) >= 0).all(), \
        "All test slots indices (argument test_slots_ind) must be postive and lower, than number of slots (argument slot_num)."
    assert type(train_slots_ind) == range and type(test_slots_ind) == range and type(validat_slots_ind), \
        f"Types of train, validation and test indices must be a range, but {type(train_slots_ind)}, {type(validat_slots_ind)} and {type(test_slots_ind)} are given correspondingly."
    assert train_slots_ind.step == 1 and validat_slots_ind.step == 1 and test_slots_ind.step == 1, \
        f"Step of indices ranges train_slots_ind, validat_slots_ind and test_slots_ind must equal 1, but {train_slots_ind.step}, {validat_slots_ind.step} and {test_slots_ind.step} are given correspondingly."

    slot_input_size = int(input.shape[-1]/slot_num)
    slot_target_size = int(target.shape[-1]/slot_num)
    
    input_train_size = slot_input_size * len(train_slots_ind)
    target_train_size = slot_target_size * len(train_slots_ind)
    input_validat_size = slot_input_size * len(validat_slots_ind)
    target_validat_size = slot_target_size * len(validat_slots_ind)
    input_test_size = slot_input_size * len(test_slots_ind)
    target_test_size = slot_target_size * len(test_slots_ind)
    
    if block_size is None: 
        block_size = input_train_size
    block_size_target = block_size
    
    if block_size is None: 
        block_size = input_test_size
    block_size_test = block_size
    block_size_test_target = block_size_test

    if block_size is None: 
        block_size = input_validat_size
    block_size_validat = block_size
    block_size_validat_target = block_size_validat
    
    dataset = list()
    
    train_input_set = input[..., train_slots_ind[0] * slot_input_size: train_slots_ind[0] * slot_input_size + input_train_size]
    train_input_set = F.pad(train_input_set, (pad_zeros, pad_zeros))
    train_target_set = target[..., train_slots_ind[0] * slot_target_size: train_slots_ind[0] * slot_target_size + target_train_size]
    train_input_set = train_input_set.reshape(1, 2, -1)
    train_target_set = train_target_set.reshape(1, 1, -1)
    train_input_set = train_input_set.unfold(2, block_size + 2*pad_zeros, int(block_size))[0, ...].permute(1, 0, 2)
    train_target_set = train_target_set.unfold(2, block_size_target, int(block_size_target))[0, ...].permute(1, 0, 2)
    train_set = tuple((train_input_set, train_target_set))
    train_set = ResampleDataset(train_set, batch_size=batch_size)
    train_set = torch.utils.data.DataLoader(train_set, batch_size=None)
    
    validat_input_set = input[..., validat_slots_ind[0] * slot_input_size: validat_slots_ind[0] * slot_input_size + input_validat_size]
    validat_input_set = F.pad(validat_input_set, (pad_zeros, pad_zeros))
    validat_target_set = target[..., validat_slots_ind[0] * slot_target_size: validat_slots_ind[0] * slot_target_size + target_validat_size]
    validat_input_set = validat_input_set.reshape(1, 2, -1)
    validat_target_set = validat_target_set.reshape(1, 1, -1)
    validat_input_set = validat_input_set.unfold(2, block_size_validat + 2*pad_zeros, block_size_validat)[0, ...].permute(1, 0, 2)
    validat_target_set = validat_target_set.unfold(2, block_size_validat_target, block_size_validat_target)[0, ...].permute(1, 0, 2)
    validat_set = tuple((validat_input_set, validat_target_set))
    validat_set = ResampleDataset(validat_set, batch_size=batch_size)
    validat_set = torch.utils.data.DataLoader(validat_set, batch_size=None)

    test_input_set = input[..., test_slots_ind[0] * slot_input_size: test_slots_ind[0] * slot_input_size + input_test_size]
    test_input_set = F.pad(test_input_set, (pad_zeros, pad_zeros))
    test_target_set = target[..., test_slots_ind[0] * slot_target_size: test_slots_ind[0] * slot_target_size + target_test_size]
    test_input_set = test_input_set.reshape(1, 2, -1)
    test_target_set = test_target_set.reshape(1, 1, -1)
    test_input_set = test_input_set.unfold(2, block_size_test + 2*pad_zeros, block_size_test)[0, ...].permute(1, 0, 2)
    test_target_set = test_target_set.unfold(2, block_size_test_target, block_size_test_target)[0, ...].permute(1, 0, 2)
    test_set = tuple((test_input_set, test_target_set))
    test_set = ResampleDataset(test_set, batch_size=batch_size)
    test_set = torch.utils.data.DataLoader(test_set, batch_size=None)
    
    dataset.append(tuple((train_set, validat_set, test_set)))
    return dataset[0]

def dataset_prepare(mat: dict, dtype: torch.dtype = torch.complex128, device: str = 'cuda', batch_size: OptionalInt = None, 
                    block_size: OptionalInt = None, slot_num: OptionalInt = None, pad_zeros: OptionalInt = None, 
                    delay_d: OptionalInt = None, train_slots_ind: range = range(1), validat_slots_ind: range = range(1),
                    test_slots_ind: range = range(1)) -> DatasetType:
    """
    The method extracts input and target data for the mat file, normalizes and resamples if necessary.
    Then it divides input and target tensors into the batches and loads them into the dataloader.

    Args:
        mat (Dictionary): mat-file, which contains:
            input (numpy.ndarray): 1d array with shape (1, arr.shape), which contains input data samples.
            target (numpy.ndarray): 1d array with shape (1, arr.shape), which contains target data samples.
            noise_floor (numpy.ndarray): 1d array with shape (1, arr.shape), which contains noise floor samples.
        dtype (torch.dtype): The type of tensor to convert content of the mat-file to. Defaults is torch.complex128.
        device (str): The device to load dataset to. Defaults is 'cpu'.
        slot_num (int, optional): The number of slots to divide the whole dataset into.
        pad_zeros (int, optional): The number of zeros to add to the beginning and to the end of the signal.
        delay_d (int, optional): The value of desired and noise floor signals shift. delay_d > 0 corresponds to
            samples shift left, delay_d < 0 corresponds to samples shift right. If delay_d equal zero or None, 
            then there is no shift. Defaults is "None".
        train_slots_ind (range): Indices of the slots which are chosen for training dataset. A range with step 1. Defaults is range(1).
        validat_slots_ind (range): Used only for hold-out cross-validation. Indices of the slots which are chosen for validation dataset. 
            A range with step 1. Defaults is range(1).
        test_slots_ind (range): Indices of the slots which are chosen for training dataset. A range with step 1. Defaults is range(1).
            
    Returns:
        Tuple of iterables.
    """
    
    if pad_zeros is None:
        pad_zeros = 0

    if batch_size is None:
        batch_size = 1

    input_a = mat['PDinA'][0, :]
    input_b = mat['PDinB'][0, :]
    target_a = mat['PDoutA'][0, :] - mat['PDinA'][0, :]
    target_b = mat['PDoutB'][0, :] - mat['PDinB'][0, :]
    nf = np.zeros_like(target_a)

    if delay_d is not None and delay_d != 0:
        target_a = np.roll(target_a, -delay_d)
        target_b = np.roll(target_b, -delay_d)
        nf = np.roll(nf, -delay_d)

    input_tens_a = torch.tensor(input_a, dtype=dtype).view(1, 1, -1).to(device)
    input_tens_b = torch.tensor(input_b, dtype=dtype).view(1, 1, -1).to(device)
    input = torch.cat((input_tens_a, input_tens_b), dim=1)
    target_tens_a = torch.tensor(target_a, dtype=dtype).view(1, 1, -1).to(device)
    target_tens_b = torch.tensor(target_b, dtype=dtype).view(1, 1, -1).to(device)
    target = torch.cat((target_tens_a, target_tens_b), dim=1)
    nf = torch.tensor(nf, dtype=dtype).view(1, 1, -1).to(device)

    input = input/30000
    target = target/30000
    # alpha = 0.9#/30000
    # scale_input_a = input[:, :1, :].abs().max()
    # scale_input_b = input[:, 1:, :].abs().max()
    # input[:, :1, :] = alpha * (input[:, :1, :].to(device) / scale_input_a)
    # input[:, 1:, :] = alpha * (input[:, 1:, :].to(device) / scale_input_b)
    # scale_target_a = target[:, :1, :].abs().max()
    # scale_target_b = target[:, 1:, :].abs().max()
    # target[:, :1, :] = alpha * (target[:, :1, :].to(device) / scale_target_a)
    # target[:, 1:, :] = alpha * (target[:, 1:, :].to(device) / scale_target_b)

    assert (np.array(train_slots_ind) < slot_num).all() and (np.array(train_slots_ind) >= 0).all(), \
        "All train slots indices (argument train_slots_ind) must be positive and lower, than number of slots (argument slot_num)."
    assert (np.array(validat_slots_ind) < slot_num).all() and (np.array(validat_slots_ind) >= 0).all(), \
        "All validation slots indices (argument validat_slots_ind) must be postive and lower, than number of slots (argument slot_num)."
    assert (np.array(test_slots_ind) < slot_num).all() and (np.array(test_slots_ind) >= 0).all(), \
        "All test slots indices (argument test_slots_ind) must be postive and lower, than number of slots (argument slot_num)."
    assert type(train_slots_ind) == range and type(test_slots_ind) == range and type(validat_slots_ind), \
        f"Types of train, validation and test indices must be a range, but {type(train_slots_ind)}, {type(validat_slots_ind)} and {type(test_slots_ind)} are given correspondingly."
    assert train_slots_ind.step == 1 and validat_slots_ind.step == 1 and test_slots_ind.step == 1, \
        f"Step of indices ranges train_slots_ind, validat_slots_ind and test_slots_ind must equal 1, but {train_slots_ind.step}, {validat_slots_ind.step} and {test_slots_ind.step} are given correspondingly."

    slot_input_size = int(input.shape[-1]/slot_num)
    slot_target_size = int(target.shape[-1]/slot_num)
    
    input_train_size = slot_input_size * len(train_slots_ind)
    target_train_size = slot_target_size * len(train_slots_ind)
    input_validat_size = slot_input_size * len(validat_slots_ind)
    target_validat_size = slot_target_size * len(validat_slots_ind)
    input_test_size = slot_input_size * len(test_slots_ind)
    target_test_size = slot_target_size * len(test_slots_ind)
    
    if block_size is None: 
        block_size = input_train_size
    block_size_target = block_size
    
    block_size_test = input_test_size
    block_size_test_target = block_size_test

    block_size_validat = input_validat_size
    block_size_validat_target = block_size_validat
    
    dataset = list()
    
    train_input_set = input[..., train_slots_ind[0] * slot_input_size: train_slots_ind[0] * slot_input_size + input_train_size]
    train_input_set = F.pad(train_input_set, (pad_zeros, pad_zeros))
    train_target_set = torch.cat((target, nf), dim=1)[..., train_slots_ind[0] * slot_target_size: train_slots_ind[0] * slot_target_size + target_train_size]

    train_input_set = train_input_set.unfold(2, block_size + 2*pad_zeros, int(block_size))[0, ...].permute(1, 0, 2)
    train_target_set = train_target_set.unfold(2, block_size_target, int(block_size_target))[0, ...].permute(1, 0, 2)
    train_set = tuple((train_input_set, train_target_set))
    train_set = ResampleDataset(train_set, batch_size=batch_size)

    train_set = torch.utils.data.DataLoader(train_set, batch_size=None)
    
    validat_input_set = input[..., validat_slots_ind[0] * slot_input_size: validat_slots_ind[0] * slot_input_size + input_validat_size]
    validat_input_set = F.pad(validat_input_set, (pad_zeros, pad_zeros))
    validat_target_set = torch.cat((target, nf), dim=1)[..., validat_slots_ind[0] * slot_target_size: validat_slots_ind[0] * slot_target_size + target_validat_size]
    validat_input_set = validat_input_set.unfold(2, block_size_validat + 2*pad_zeros, block_size_validat)[0, ...].permute(1, 0, 2)
    validat_target_set = validat_target_set.unfold(2, block_size_validat_target, block_size_validat_target)[0, ...].permute(1, 0, 2)
    validat_set = tuple((validat_input_set, validat_target_set))
    validat_set = ResampleDataset(validat_set, batch_size=batch_size)
    validat_set = torch.utils.data.DataLoader(validat_set, batch_size=None)

    test_input_set = input[..., test_slots_ind[0] * slot_input_size: test_slots_ind[0] * slot_input_size + input_test_size]
    test_input_set = F.pad(test_input_set, (pad_zeros, pad_zeros))
    test_target_set = torch.cat((target, nf), dim=1)[..., test_slots_ind[0] * slot_target_size: test_slots_ind[0] * slot_target_size + target_test_size]
    test_input_set = test_input_set.unfold(2, block_size_test + 2*pad_zeros, block_size_test)[0, ...].permute(1, 0, 2)
    test_target_set = test_target_set.unfold(2, block_size_test_target, block_size_test_target)[0, ...].permute(1, 0, 2)
    test_set = tuple((test_input_set, test_target_set))
    test_set = ResampleDataset(test_set, batch_size=batch_size)
    test_set = torch.utils.data.DataLoader(test_set, batch_size=None)
    
    dataset.append(tuple((train_set, validat_set, test_set)))
    return dataset[0]