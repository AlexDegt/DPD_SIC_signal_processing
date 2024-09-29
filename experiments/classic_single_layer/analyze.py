import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
from copy import deepcopy
from scipy.io import loadmat, savemat

def nmse_fn(x, e):
    """ Returns Normalized Mean Squared error """
    y = 10.0*np.log10(np.real((np.sum(e*np.conj(e))/np.sum(x*np.conj(x)))))
    return y

def aclr_fn(sig, f, fs=1.0, nfft=1024, window='blackman', nperseg=None, noverlap=None):
    """ 
        Calculate Adjacent Channel Leakage Ratio
    """
    win = signal.get_window(window, nfft, True)
    freqs, psd = signal.welch(sig, fs, win, return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    ind1 = (nfft // 2) - int(np.ceil(nfft * f/fs))
    ind2 = (nfft // 2) + int(np.ceil(nfft * f/fs))
    aclr = np.sum(psd[ind1: ind2])/(np.sum(psd[:ind1]) + np.sum(psd[ind2:]))
    return 10 * np.log10(aclr)

add_folder = os.path.join("one_dim")
# add_folder = os.path.join("three_dim")
# add_folder = os.path.join("six_dim")
# add_folder = os.path.join("nine_dim")
curr_path = os.getcwd()
load_path = os.path.join(curr_path, add_folder)

f = 10 # MHz
fs = 245.76 # MHz
nfft = 1024

power_cases = [-15, -12, -9, -6, -3, 0]

param_num = list(np.arange(2, 22, 2))
# param_num = list(np.arange(4, 20, 4))

nmse, aclr = [], []

for j, param in enumerate(param_num):

    tx = np.load(os.path.join(load_path, f"{param}_param_4_slot_6_cases", "x.npy"))
    pa_out = np.load(os.path.join(load_path, f"{param}_param_4_slot_6_cases", "d.npy"))
    model_out = np.load(os.path.join(load_path, f"{param}_param_4_slot_6_cases", "y.npy"))
    sig_len = int(len(tx) / len(power_cases))

    curr_case_nmse, curr_case_aclr = [], []
    print(f"Parameter number: {param}")
    for i in range(len(power_cases)):
        x = tx[i * sig_len: (i + 1) * sig_len]
        d = pa_out[i * sig_len: (i + 1) * sig_len]
        y = model_out[i * sig_len: (i + 1) * sig_len]

        nmse_val = nmse_fn(d, d - y)
        aclr_val = aclr_fn(x + d - y, f, fs, nfft)
        curr_case_nmse.append(nmse_val)
        curr_case_aclr.append(aclr_val)

        print(f"Case {power_cases[i]} dB: NMSE = {nmse_val} dB, ACLR = {aclr_val} dB")
        plt.figure(i + 1)
        plt.title(f"Power case {power_cases[i]} dB")
    nmse.append(curr_case_nmse)
    aclr.append(curr_case_aclr)

np.save(os.path.join(load_path, "nmse.npy"), np.array(nmse))
np.save(os.path.join(load_path, "aclr.npy"), np.array(aclr))