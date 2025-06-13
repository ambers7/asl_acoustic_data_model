import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from math import ceil
'''adds padding to end'''
def collate_various_size_end(batch):
    # print("collate_various_size1111")
    data_list_arr = [x[0] for x in batch]
    target = [x[1] for x in batch]
    # Find max height and max time length
    max_height = max([x.shape[1] for x in data_list_arr])
    max_time = max([x.shape[2] for x in data_list_arr])
    window_size = 10
    padded_height = ceil(max_height / window_size) * window_size
    padded_time = ceil(max_time / window_size) * window_size

    data_arr = np.zeros((len(batch), data_list_arr[0].shape[0], padded_height, padded_time), dtype=np.float32)
    for i, arr in enumerate(data_list_arr):
        h, t = arr.shape[1], arr.shape[2]
        # change padding to the end
        start_h = 0
        start_t = 0
        data_arr[i, :, start_h:start_h + h, start_t:start_t + t] = arr.numpy() if isinstance(arr, torch.Tensor) else arr

    return torch.tensor(data_arr, dtype=torch.float32), torch.tensor(target)

'''added padding to beginning adn end (of random sizes) '''
def collate_various_size(batch):
    # print("collate_various_size1111")
    data_list_arr = [x[0] for x in batch]
    target = [x[1] for x in batch]
    # Find max height and max time length
    max_height = max([x.shape[1] for x in data_list_arr])
    max_time = max([x.shape[2] for x in data_list_arr])
    window_size = 10
    padded_height = ceil(max_height / window_size) * window_size
    padded_time = ceil(max_time / window_size) * window_size

    data_arr = np.zeros((len(batch), data_list_arr[0].shape[0], padded_height, padded_time), dtype=np.float32)
    for i, arr in enumerate(data_list_arr):
        h, t = arr.shape[1], arr.shape[2]
        pad_h = padded_height - h
        pad_t = padded_time - t
        # Randomly shift in both axes if you want augmentation, or just pad at the start
        start_h = np.random.randint(0, pad_h + 1) if pad_h > 0 else 0
        start_t = np.random.randint(0, pad_t + 1) if pad_t > 0 else 0
        data_arr[i, :, start_h:start_h + h, start_t:start_t + t] = arr.numpy() if isinstance(arr, torch.Tensor) else arr

    return torch.tensor(data_arr, dtype=torch.float32), torch.tensor(target)

'''original in emo_sign_cnn_gan.ipynb'''
def collate_various_size_original(batch):
    print("collate_various_size1111")
    data_list_arr = [x[0] for x in batch]
    target = [x[1] for x in batch]
    data_max_size = max([x.shape[1] for x in data_list_arr])
    max_time = max([x.shape[2] for x in data_list_arr])

        
    # check the windown size, for example, if windion size 10, the target size should be dividied by windon size. 
    #target_length = ceil(target_length / 16) * 16
    window_size = 10
    target_length = data_max_size 
    target_length = ceil(target_length / window_size) * window_size


    data_arr = np.zeros((len(batch), data_list_arr[0].shape[0], target_length, data_list_arr[0].shape[2]))
    # horizontal shifting time axis. 
    for i in range(0, len(data_list_arr)):
        pad_space = target_length - data_list_arr[i].shape[1]
        if pad_space > 0:
            start_x = np.random.randint(0, pad_space + 1)
        else:
            start_x = 0  # No padding needed, or sample is already at target length
        data_arr[i, :, start_x: start_x + data_list_arr[i].shape[1], :] = data_list_arr[i]