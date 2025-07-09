from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from cgi import test
from pprint import pprint
from libs.utils import (AverageMeter, save_checkpoint, IntervalSampler,
                        create_optim, create_scheduler, print_and_log, save_gt, generate_cm, extract_labels)
from libs.models import EncoderDecoder as ModelBuilder
from libs.models import Point_dis_loss, calculate_dis, get_criterion, wer_sliding_window
from libs.core import load_config
from libs.dataset import generate_data  # , gen_test
import torch.multiprocessing as mp
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import random
import numpy as np
import pandas as pd
import logging


# python imports
import argparse
import os
import re
import time
import math
import cv2
import pickle
import torch.optim as optim
from torch.nn import functional as F

from copy import deepcopy
import os
import cv2
import json
import pickle
import random
import numpy as np
from libs.dataset.data_splitter import DataSplitter
from libs.utils import print_and_log, load_gt, plot_profiles
import matplotlib.pyplot as plt


import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

def upsample_imu_data(time, imu_data, target_num_samples):
    """
    Upsample IMU data to a target number of samples.

    Parameters:
    - time: 1D array, timestamps of the original IMU data.
    - imu_data: 2D array, IMU data (e.g., acceleration, angular velocity).
    - target_num_samples: desired number of samples after upsampling.

    Returns:
    - upsampled_time: 1D array, timestamps of the upsampled data.
    - upsampled_imu_data: 2D array, upsampled IMU data.
    """
    # Ensure time values are strictly increasing and remove duplicates
    unique_time, unique_idx = np.unique(time, return_index=True)
    sorted_idx = np.argsort(unique_time)
    unique_time = unique_time[sorted_idx]
    unique_idx = unique_idx[sorted_idx]

    # Sort imu_data based on unique_time
    sorted_imu_data = imu_data[unique_idx]

    # Create an interpolation function for each dimension of the IMU data
    interp_functions = [CubicSpline(unique_time, sorted_imu_data[:, i]) for i in range(sorted_imu_data.shape[1])]
    #interp_functions = [CubicSpline(unique_time, sorted_imu_data[:, i]) for i in range(sorted_imu_data.shape[1])]

    # Create upsampled time array
    upsampled_time = np.linspace(unique_time[0], unique_time[-1], target_num_samples)

    # Interpolate IMU data at upsampled time points
    upsampled_imu_data = np.column_stack([f(upsampled_time) for f in interp_functions])

    return upsampled_time, upsampled_imu_data

# def upsample_imu_data(time, imu_data, num_upsampled_points, start_time, end_time):
#     """
#     Upsample IMU data to a specified number of data points within a given time range.

#     Parameters:
#     - time: 1D array, timestamps of the original IMU data.
#     - imu_data: 2D array, IMU data (e.g., acceleration, angular velocity).
#     - num_upsampled_points: desired number of data points after upsampling.
#     - start_time: start time of the upsampled data.
#     - end_time: end time of the upsampled data.

#     Returns:
#     - upsampled_time: 1D array, timestamps of the upsampled data.
#     - upsampled_imu_data: 2D array, upsampled IMU data.
#     """
#     # Create an interpolation function for each dimension of the IMU data
#     interp_functions = [CubicSpline(time, imu_data[:, i]) for i in range(imu_data.shape[1])]

#     # Create upsampled time array
#     upsampled_time = np.linspace(start_time, end_time, num_upsampled_points)

#     # Interpolate IMU data at upsampled time points
#     upsampled_imu_data = np.column_stack([f(upsampled_time) for f in interp_functions])

#     return upsampled_time, upsampled_imu_data

def normalize_imu_data(upsampled_imu_data):
    """
    Normalize upsampled IMU data.

    Parameters:
    - upsampled_imu_data: 2D array, upsampled IMU data.

    Returns:
    - normalized_imu_data: 2D array, normalized IMU data.
    - means: 1D array, means of each axis before normalization.
    - stds: 1D array, standard deviations of each axis before normalization.
    """
    means = np.mean(upsampled_imu_data, axis=0)
    stds = np.std(upsampled_imu_data, axis=0)

    normalized_imu_data = (upsampled_imu_data - means) / stds

    return normalized_imu_data, means, stds


def check_data_integrity(data_piece):
    # print(np.sum(np.sum(np.abs(data_piece) < 1e-3, axis=1) == data_piece.shape[1]))
    return True or (np.sum(np.sum(np.abs(data_piece) < 1e-3, axis=1) == data_piece.shape[1]) == 0)

def filter_truth(all_truth, target_length, is_train=False):
    filtered_truth = []
    # last_truth = None
    # last_last_truth = None
    for i in range(len(all_truth)):
        if isinstance(all_truth[i][0], str) and len(all_truth[i][0].split()) == 1 and int(all_truth[i][0]) < 0:
            continue
        # if isinstance(all_truth[i][0], str) and (len(all_truth[i][0].split()) > 1):# or int(all_truth[i][0]) >= 10:
        #     continue
        # if int(all_truth[i][0]) > 0:
        #     all_truth[i] = ('', all_truth[i][1], all_truth[i][2], '')
        # else:
        #     all_truth[i] = ('0', all_truth[i][1], all_truth[i][2], 'EchoGlass')
        # if isinstance(all_truth[i][0], str) and len(all_truth[i][0].split()) == 1 and (not 10 <= int(all_truth[i][0]) <= 16):
        #     continue
        
        # only music player commands
        commands_maps = {
            '10': '0',
            '11': '1',
            '12': '2',
            '13': '3',
            '14': '4',
            '15': '5',
            '16': '6',
            '19': '7',
            '20': '8',
        }
        # if isinstance(all_truth[i][0], str) and len(all_truth[i][0].split()) == 1 and all_truth[i][0] in commands_maps:
        #     all_truth[i] = (commands_maps[all_truth[i][0]], all_truth[i][1], all_truth[i][2], all_truth[i][3])
        # else:
        #     continue

        # if all_truth[i][0] == '25':
        #     all_truth[i] = ('25 19', all_truth[i][1], all_truth[i][2], all_truth[i][3])
        # if all_truth[i][0] == '28':
        #     all_truth[i] = ('31 28', all_truth[i][1], all_truth[i][2], all_truth[i][3])
        # if all_truth[i][0] == '29':
        #     all_truth[i] = ('31 29', all_truth[i][1], all_truth[i][2], all_truth[i][3])

        
        # if all_truth[i][3] == 'Volume up':
        #     all_truth[i] = ('6 7', all_truth[i][1], all_truth[i][2], all_truth[i][3])
        # if all_truth[i][3] == 'Volume down':
        #     all_truth[i] = ('6 8', all_truth[i][1], all_truth[i][2], all_truth[i][3])
        # if all_truth[i][3][0] == 'a':
        #     continue
        # if not is_train:
        filtered_truth += [all_truth[i]]
        if is_train and 1:
            cur_start = float(all_truth[i][1])
            pre = i - 1
            combined_start = float(all_truth[i][1])
            combined_end = float(all_truth[i][2])
            combined_labels = [all_truth[i][0]]
            combined_texts = [all_truth[i][3]]
            while pre >= 0 and cur_start - float(all_truth[pre][2]) < 0.3 and combined_end - float(all_truth[pre][1]) < target_length:
            # while pre >= 0 and cur_start - float(all_truth[pre][2]) < 0.3 and i - pre < 3:
                combined_labels += [all_truth[pre][0]]
                combined_texts += [all_truth[pre][3]]
                combined_start = float(all_truth[pre][1])
                if len(combined_labels) > 1:
                    filtered_truth += [(' '.join(combined_labels[::-1]), combined_start, combined_end, ' '.join(combined_texts[::-1]))]
                cur_start = combined_start
                pre -= 1
        # if is_train and last_truth is not None and float(truth[1]) - float(last_truth[2]) < 0.5:
        #     combined_label = last_truth[0] + ' ' + truth[0]
        #     combined_text = last_truth[3] + ' ' + truth[3]
        #     filtered_truth += [(combined_label, last_truth[1], truth[2], combined_text)]
        #     if last_last_truth is not None and float(last_truth[1]) - float(last_last_truth[2]) < 0.5:
        #         combined_label = last_last_truth[0] + ' ' + last_truth[0] + ' ' + truth[0]
        #         combined_text = last_last_truth[3] + ' ' + last_truth[3] + ' ' + truth[3]
        #         filtered_truth += [(combined_label, last_last_truth[1], truth[2], combined_text)]
        # # last_last_truth = deepcopy(last_truth)
        # last_truth = deepcopy(truth)
    return filtered_truth
def plot_data(input_arr, input_imu, target, batch):
    i =batch
    ITOS = {0: 'a',
                    1: 'b',
                    2: 'c',
                    3: 'd',
                    4: 'e',
                    5: 'f',
                    6: 'g',
                    7: 'h',
                    8: 'i',
                    9: 'j',
                    10: 'k',
                    11: 'l',
                    12: 'm',
                    13: 'n',
                    14: 'o',
                    15: 'p',
                    16: 'q',
                    17: 'r',
                    18: 's',
                    19: 't',
                    20: 'u',
                    21: 'v',
                    22: 'w',
                    23: 'x',
                    24: 'y',
                    25: 'z', 
                    26: '^'}
    for kk in range(0, len(input_arr)):
        label_nam = ''.join([ITOS[int(i)] for i in target[kk].split()])
        if len(label_nam) < 13:
            img = input_arr[kk].reshape(input_arr[kk].shape[1], input_arr[kk].shape[2]).numpy()
            echo_profile_img_bgr = plot_profiles(img.T, 10000000, -10000000)
            # bgr to rgb
            echo_profile_img = echo_profile_img_bgr.copy()
            echo_profile_img[:,:,0] = echo_profile_img_bgr[:,:,2]
            echo_profile_img[:,:,2] = echo_profile_img_bgr[:,:,0]
            echo_profile_img = echo_profile_img.reshape(echo_profile_img.shape[1], echo_profile_img.shape[0], echo_profile_img.shape[2])
            
            plt.figure(1)
            plt.subplot(211)
            #plt.figure(figsize = (20,2))
            plt.imshow(echo_profile_img_bgr)
            plt.xticks([])
            label_nam = ''.join([ITOS[int(i)] for i in target[kk].split()])
            print([ITOS[int(i)] for i in target[kk].split()])
            plt.title([ITOS[int(i)] for i in target[kk].split()])
            plt.subplot(212)
            print(input_imu[kk].shape)
            imu = input_imu[kk][0].numpy()
            imu = imu.reshape(imu.shape[1],imu.shape[0])
            print(imu.shape)
            plt.plot(imu[0].T, )
            plt.plot(imu[1].T)
            plt.plot(imu[2].T)
            ee = [i for i in range(len(imu[0].T)) if i %16 == 0]

            # plt.vlines(x = ee, ymin = 0, ymax = max(imu[0].T),
            # colors = 'purple',
            # linestyle = '-',
            # label = 'vline_multiple - full height')
            plt.xlim(2, len(imu[0].T))
            plt.tight_layout()
            plt.legend(['x', 'y', 'z'])
            paths = './images/%s'%(args.output.split('/')[-1])
            if os.path.isdir(paths) is not True:
                os.mkdir(paths)

            plt.savefig('./images/%s/img%d_%d_%s.png'%(args.output.split('/')[-1], i, kk, label_nam))
            plt.cla()

def get_sessions(dataset_path):
    return [f for f in os.listdir(dataset_path) if not f.startswith('.')]

def plot_profiles(profiles, max_val=20000000, min_val=-20000000):
    max_h = 0       # red
    min_h = 120     # blue
    if not max_val:
        max_val = np.max(profiles)
    if not min_val:
        min_val = np.min(profiles)
    # print(max_val, min_val)
    heat_map_val = np.clip(profiles, min_val, max_val)
    heat_map = np.zeros(
        (heat_map_val.shape[0], heat_map_val.shape[1], 3), dtype=np.uint8)
    # print(heat_map_val.shape)
    heat_map[:, :, 0] = heat_map_val / \
        (max_val + 1e-6) * (max_h - min_h) + min_h
    heat_map[:, :, 1] = np.ones(heat_map_val.shape) * 255
    heat_map[:, :, 2] = np.ones(heat_map_val.shape) * 255
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_HSV2BGR)
    return heat_map

def plot_profiles_split_channels(profiles, n_channels, maxval=None, minval=None):
    channel_width = profiles.shape[0] // n_channels

    profiles_img = np.zeros(
        ((channel_width + 5) * n_channels, profiles.shape[1], 3))

    for n in range(n_channels):
        channel_profiles = profiles[n * channel_width: (n + 1) * channel_width]
        profiles_img[n * (channel_width + 5): (n + 1) * (channel_width + 5) - 5,
                     :, :] = plot_profiles(channel_profiles, maxval, minval)

    return profiles_img



def read_from_folder1(session_num, data_path, is_train=False):
    # data_pairs = []
    # session_config = json.load(open(config_file, 'rt'))
    # # print("data_file: ", data_file)
    # audio_config = session_config['audio_config']

    # audio_syncing_sample = session_config['syncing']['audio']
    # gt_syncing_ts = session_config['syncing']['ground_truth']
    # response_offset = 0.2
    # if 'response_offset' in session_config:
    #     response_offset = session_config['response_offset']
    # # print(profile_syncing_idx, gt_syncing_ts)
    # def gt_ts_to_profile_idx(ts, idx_offset=0):
    #     return round(((ts - gt_syncing_ts) * audio_config['sampling_rate'] + audio_syncing_sample) / audio_config['frame_length']) - idx_offset
        
    # all_profiles = np.load(data_file)
    # #all_original_profiles = np.load(data_file.replace('_diff', ''))[:, 1:]
    # #all_profiles = np.concatenate([all_profiles, all_original_profiles], axis=0)
    # # print("data_save: ", all_profiles.shape)
    

    # target_length = input_config['train_target_length'] if is_train else input_config['test_target_length']

    # #### IMU
    # imu_config = data_file.split('/fmcw')[0]+'/imu.txt'
    # imu_gnd_config = data_file.split('/fmcw')[0]+'/ground_truth_classification_imu.txt'
    # # print('----------', imu_config)
    # File_data = np.loadtxt(imu_config, dtype=str, delimiter=",") 
    # all_raw = np.array(File_data, dtype=float)
    # all_imu = np.array(File_data, dtype=float)[:, 1:]
    # all_imu_time = np.array(File_data, dtype=float)[:, :1]

    # File_data = np.loadtxt(imu_gnd_config, dtype=str, delimiter=",") 
    # all_raw_gnd = np.array(File_data)
    # all_imu_gnd = np.array(File_data)[:, 4:6]
    # #print(all_imu_gnd)
    # #print(max([int(i[1]) - int(i[0]) for i in all_imu_gnd]))
    # max_length_imu = max([int(i[1]) - int(i[0]) for i in all_imu_gnd])
    # cnt =0

    # # ### tmp ratio across participants #####
    # # r = np.load('/data/ruidong/acoustic_silentspeech/dataset/pilot_study/0711_ke_speechin/r7.npy')
    # # all_profiles = all_profiles[:512, :]
    # # all_profiles *= np.repeat(r[:, None], all_profiles.shape[1], axis=1)
    # # mu, sigma = np.mean(all_profiles), np.std(all_profiles)
    # # print(mu, sigma)
    # # all_profiles = (all_profiles - mu) / sigma
    # all_profiles = all_profiles.astype('float32').T      # size n x (1023 * channels)
    # if input_config['remove_static']:
    #     static_profiles = np.load(static_file).astype('float32').T       # size n x (1023 * channels)
    #     print_and_log('Static path found at %s' % static_file)
    #     assert(static_profiles.shape == all_profiles.shape)                   # exact same size
    # # all_profiles = np.abs(all_profiles)

    # if isinstance(audio_config['tx_file'], str):
    #     n_tx = 1
    # else:
    #     n_tx = len(audio_config['tx_file'])
    # n_channels_prepared = len(audio_config['channels_of_interest']) * n_tx
    # input_config['n_channels_prepared'] = n_channels_prepared
    # if len(input_config['channels_of_interest']) == 0:
    #     input_config['channels_of_interest'] = list(range(n_channels_prepared))
    # input_config['n_channels_loaded'] = len(input_config['channels_of_interest'])
    # assert(all_profiles.shape[1] % n_channels_prepared == 0) # data size must match the channels
    # profile_channel_height = all_profiles.shape[1] // n_channels_prepared
    # poi = input_config['pixels_of_interest']
    # if poi != (0, 0):
    #     columns_of_interest = []
    #     for i in input_config['channels_of_interest']:
    #         columns_of_interest += list(range(profile_channel_height * i + poi[0] + profile_channel_height // 2, profile_channel_height * i + poi[1] + profile_channel_height // 2))
    #     all_profiles = all_profiles[:, columns_of_interest]
    #     if input_config['remove_static']:
    #         static_profiles = static_profiles[:, columns_of_interest]

    # print("data_save111: ", all_profiles.shape)
    # # print(columns_of_interest)
    # all_truth = load_gt(truth_file)
    # all_truth = filter_truth(all_truth, target_length=(target_length * audio_config['frame_length'] / audio_config['sampling_rate']), is_train=is_train)
    # # print([t[0] for t in all_truth])
    # # all_truth[:, 1:] *= 1000
    # if truth_file[-4:] == '.npy':
    #     all_truth_idx = [gt_ts_to_profile_idx(t[0]) for t in all_truth]
    # else:
    #     all_truth_idx = [gt_ts_to_profile_idx(float(t[1]) + response_offset - 2 * input_config['h_shift']) for t in all_truth] + [gt_ts_to_profile_idx(float(t[2]) + response_offset + 2 * input_config['h_shift']) for t in all_truth]
    # affine_offset = (0, 0)
    # if is_train and input_config['augment_affine']:
    #     affine_offset = input_config['affine_parameters']['move'][0]
    #     affine_added_length = affine_offset[1] - affine_offset[0]
    # all_profiles = all_profiles[min(all_truth_idx) + affine_offset[0]: max(all_truth_idx) + 1 + affine_offset[1], :]  # remove profile of other sessions, they are useless
    # # mu, sigma = np.mean(all_profiles), np.std(all_profiles)
    # # all_profiles = (all_profiles - mu) / sigma
    # # print(mu, sigma)
    # # print(min(all_truth_idx), max(all_truth_idx))
    # if input_config['remove_static']:
    #     static_profiles = static_profiles[min(all_truth_idx) + affine_offset[0]: max(all_truth_idx) + 1 + affine_offset[1], :]  # remove profile of other sessions, they are useless
    #     # static_profiles = (static_profiles - mu) / sigma

    # # all_profiles = cv2.resize(all_profiles, (all_profiles.shape[1] // 3, all_profiles.shape[0]))
    # remove_offset = min(all_truth_idx) + affine_offset[0]
    # loaded_gt = []
    # n_bad = 0
    # static_profiles_data_piece = None
    # actual_lens = []
    # input_config['h_shift_frames'] = round(input_config['h_shift'] * audio_config['sampling_rate'] / audio_config['frame_length'])
    # # print(input_config['h_shift_frames'])
    # if (not is_train) and input_config['test_sliding_window']['applied']:
    #     for profile_s_ind in range(0, all_profiles.shape[0] - target_length, input_config['test_sliding_window']['stride']):
    #         profile_e_ind = profile_s_ind + target_length
    #         actual_lens += [profile_e_ind - profile_s_ind]
    #         # TODO: detect and discard bad pieces
    #         piece_fake_gt = (profile_s_ind + index_offset, profile_e_ind + index_offset)
    #         # print(piece_fake_gt)
    #         profile_data_piece = all_profiles[profile_s_ind: profile_e_ind, :]
    #         if input_config['remove_static']:
    #             static_profiles_data_piece = (static_profiles[profile_s_ind, :] + static_profiles[profile_e_ind, :]) / 2
    #         if input_config['stacking'] == 'channel':
    #             profile_data_piece = profile_data_piece.T
    #             profile_data_piece.shape = len(input_config['channels_of_interest']), -1, profile_data_piece.shape[1]
    #             profile_data_piece = profile_data_piece.swapaxes(1, 2)
    #             if input_config['remove_static']:
    #                 static_profiles_data_piece = static_profiles_data_piece.T
    #                 static_profiles_data_piece.shape = len(input_config['channels_of_interest']), -1, static_profiles_data_piece.shape[1]
    #                 static_profiles_data_piece = static_profiles_data_piece.swapaxes(1, 2)
    #         else:
    #             profile_data_piece = profile_data_piece[None, ...]
    #             if input_config['remove_static']:
    #                 static_profiles_data_piece = static_profiles_data_piece[None, ...]
    #         data_pairs += [(profile_data_piece, static_profiles_data_piece, piece_fake_gt)]
    #     for truth in all_truth:
    #         truth_s_ind = gt_ts_to_profile_idx(float(truth[1]) + response_offset, remove_offset) + index_offset
    #         truth_e_ind = gt_ts_to_profile_idx(float(truth[2]) + response_offset, remove_offset) + index_offset
    #         loaded_gt += [(truth[0], truth[1], truth[2], truth[3], truth_s_ind, truth_e_ind)]
    # else:
    #     for truth in all_truth:
    #         if truth_file[-4:] == '.npy':
    #             profile_e_ind = gt_ts_to_profile_idx(truth[0], remove_offset) + target_length - 1
    #             profile_s_ind = profile_e_ind - target_length
    #             if profile_s_ind < 0 or profile_e_ind >= all_profiles.shape[0]:
    #                 continue
    #             # truth_data = truth[1:41].astype(np.float32) / 17.56
    #             truth_data = truth[1:].astype(np.float32)
    #         else:
    #             profile_s_ind = gt_ts_to_profile_idx(float(truth[1]) + response_offset - input_config['h_shift'], remove_offset)
    #             profile_e_ind = gt_ts_to_profile_idx(float(truth[2]) + response_offset + input_config['h_shift'], remove_offset)
    #             # truth = (str(int(truth[0]) - 8), truth[1], truth[2], truth[3])
    #             truth_data = truth[0]
    #         actual_lens += [profile_e_ind - profile_s_ind]
    #         profile_data_piece = all_profiles[profile_s_ind: profile_e_ind, :]
    #         #print("raw:", profile_data_piece.shape)
    #         if input_config['remove_static']:
    #             static_profiles_data_piece = (static_profiles[profile_s_ind, :] + static_profiles[profile_e_ind, :]) / 2
    #         # if profile_data_piece.shape[0] != profile_dp_left_length + profile_dp_right_length + 1:
    #         #     continue
    #         if not check_data_integrity(profile_data_piece):
    #             n_bad += 1
    #             continue
    #         if profile_data_piece.shape[0] < 10:
    #             print('Warning: too short', data_file, truth)
    #         if np.std(profile_data_piece) < 1e-3:
    #             print('Warning: empty', data_file, truth)
    #         # if profile_data_piece.shape[0] > target_length + 2 * input_config['h_shift_frames'] > 10:
    #         #     print('Warning: skipped')
    #         #     continue
    #         # if truth[0] == '5':
    #         #     truth = ('5 6', truth[1], truth[2], truth[3])
    #         loaded_gt += [truth]
    #         # ########### debugging output ################
    #         # if int(truth_data) == 29:
    #         #     hm = plot_profiles(profile_data_piece.T, 200000000, -200000000)
    #         #     cv2.imwrite('data_piece.png', hm)
    #         #     print(truth)
    #         #     input()
    #         # #############################################
    #         # if profile_data_piece.shape[0] < target_length + affine_added_length:
    #         #     # print(profile_data_piece.shape[0], truth)
    #         #     # zero padding
    #         #     padding_left = np.zeros((round((target_length - profile_data_piece.shape[0]) / 2 + 0.5 + affine_offset[0]), profile_data_piece.shape[1]), profile_data_piece.dtype)
    #         #     padding_right = np.zeros((round((target_length - profile_data_piece.shape[0]) / 2 + 0.5 + affine_offset[1]), profile_data_piece.shape[1]), profile_data_piece.dtype)
    #         #     profile_data_piece = np.r_[padding_left, profile_data_piece, padding_right]
    #         # profile_data_piece = cv2.resize(profile_data_piece, (profile_data_piece.shape[1] * 2, profile_data_piece.shape[0]))
    #         if input_config['stacking'] == 'channel':
    #             print('static check')
    #             profile_data_piece = profile_data_piece.T
    #             profile_data_piece.shape = len(input_config['channels_of_interest']), -1, profile_data_piece.shape[1]
    #             profile_data_piece = profile_data_piece.swapaxes(1, 2)
    #             if input_config['remove_static']:
    #                 static_profiles_data_piece = static_profiles_data_piece.T
    #                 static_profiles_data_piece.shape = len(input_config['channels_of_interest']), -1, static_profiles_data_piece.shape[1]
    #                 static_profiles_data_piece = static_profiles_data_piece.swapaxes(1, 2)
    #         else:
    #             print('static no check')
    #             profile_data_piece = profile_data_piece[None, ...]
    #             if input_config['remove_static']:
    #                 static_profiles_data_piece = static_profiles_data_piece[None, ...]

    #         ################ IMU ############
    #         imu_s_ind = int(truth[4]) +30
    #         imu_e_ind = int(truth[5]) +30
    #         actual_length_imu = imu_e_ind - imu_s_ind + 1
    #         imu_data_piece_org = all_imu[imu_s_ind:imu_e_ind, :]
    #         time_data_piece = all_imu_time[imu_s_ind:imu_e_ind:, :]
    #         time_data_piece = np.array([i[0] for i in time_data_piece])
    #         #print("LOG: ", time_data_piece.shape, imu_data_piece_org.shape, imu_data_piece_org)
    #         # cnt+=1
    #         # path = '/data3/hyunchul/asl/Headset_silentspeech1/dl_model/image_test/%s_%s'%(data_file.split('/')[3],data_file.split('/')[5])

    #         # plt.plot(imu_data_piece_org)
    #         # plt.savefig(path + '/%s.png'%truth[3])
    #         # plt.cla()
    #         upsampled_time, upsampled_imu_data = upsample_imu_data(time_data_piece, imu_data_piece_org, profile_data_piece.shape[1])
            
    #         # plt.plot(upsampled_imu_data)
    #         # plt.savefig(path + '/%s_up.png'%truth[3])
    #         # plt.cla()

    #         normalized_imu_data, means, stds = normalize_imu_data(upsampled_imu_data)

    #         # plt.plot(normalized_imu_data)
    #         # plt.savefig(path + '/%s_up_norm.png'%truth[3])
    #         # plt.cla()

            

    #             #train_data += data_pairs
    #         if (0):
    #             # data_file.split('/fmcw')[0]
    #             path = '/data3/hyunchul/asl/Headset_silentspeech1/dl_model/image_test/%s_%s'%(data_file.split('/')[3],data_file.split('/')[5])
    #             if os.path.exists(path):
    #                 if data_file.split('/')[5] =='session_1001':
    #                     if (len(os.listdir(path)) < 117):
    #                         #print(time_data_piece)
    #                         print(data_file, profile_data_piece.shape, imu_data_piece_org.shape, truth, data_file.split('/'), path)
    #                         #print(imu_data_piece_org)
    #                         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    #                         ax1.plot(upsampled_imu_data)
    #                         ax1.set_title(
    #                                       truth[3] + ' - '
    #                                       + str((float(truth[7]) - float(truth[6]))/1000) 
    #                                       +' (sec)')                            
    #                         img = profile_data_piece.reshape(profile_data_piece.shape[1], profile_data_piece.shape[2])
    #                         echo_profile_img_bgr = plot_profiles(img.T, 10000000, -10000000)
    #                         echo_profile_img = echo_profile_img_bgr.copy()
    #                         echo_profile_img[:,:,0] = echo_profile_img_bgr[:,:,2]
    #                         echo_profile_img[:,:,2] = echo_profile_img_bgr[:,:,0]
                            
    #                         ax2.imshow(echo_profile_img,aspect="auto")
    #                         # ax2.set_title(str(float(truth[2]) - float(truth[1])))
    #                         ax2.set_xticks([])
    #                         ax2.set_yticks([])
    #                         ax2.set_xlabel(data_file)
    #                         #ax2.axhline(y=50, color='black', linestyle='-')
    #                         fig.savefig(path + '/0_%s.png'%truth[3])
    #                         #fig.savefig(path + '/%s_%d.png'%(truth[3],cnt))
    #                         fig.clear()
    #                 else:
    #                     if (len(os.listdir(path)) < 121):
    #                         #print(time_data_piece)
    #                         print(data_file, profile_data_piece.shape, imu_data_piece_org.shape, truth, data_file.split('/'), path)
    #                         #print(imu_data_piece_org)
    #                         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    #                         ax1.plot(upsampled_imu_data)
    #                         ax1.set_title(truth[3] + ' - '
    #                                       + str((float(truth[7]) - float(truth[6]))/1000) 
    #                                       +' (sec)')
    #                         img = profile_data_piece.reshape(profile_data_piece.shape[1], profile_data_piece.shape[2])
    #                         np.save(path + '/acs_%s.npy'%truth[3], img)
    #                         np.save(path + '/imu_%s.npy'%truth[3], upsampled_imu_data)
    #                         print(img)
    #                         echo_profile_img_bgr = plot_profiles(img.T, 10000000, -10000000)
    #                         echo_profile_img = echo_profile_img_bgr.copy()
    #                         echo_profile_img[:,:,0] = echo_profile_img_bgr[:,:,2]
    #                         echo_profile_img[:,:,2] = echo_profile_img_bgr[:,:,0]
    #                         ax2.imshow(echo_profile_img,aspect="auto")
    #                         #ax2.set_title(round(str(float(truth[2]) - float(truth[1])), 4))
    #                         ax2.set_xticks([])
    #                         ax2.set_yticks([])
    #                         ax2.set_xlabel(data_file)
    #                         #ax2.axhline(y=50, color='black', linestyle='-')
    #                         fig.savefig(path + '/0_%s.png'%truth[3])
    #                         #fig.savefig(path + '/%s_%d.png'%(truth[3],cnt))
    #                         fig.clear()
    #             else:
    #                 os.mkdir(path)
                
            
    #         #plt.plot(normalized_imu_data)
            
    #         # fig.savefig('/data3/hyunchul/asl/Headset_silentspeech1/dl_model/images/sessions1/%s_imu_echo.png'%truth[3])
    #         # plt.cla()
    #         # img = profile_data_piece.reshape(profile_data_piece.shape[1], profile_data_piece.shape[2])
    #         # echo_profile_img_bgr = plot_profiles(img.T, 10000000, -10000000)
    #         # echo_profile_img = echo_profile_img_bgr.copy()
    #         # echo_profile_img[:,:,0] = echo_profile_img_bgr[:,:,2]
    #         # echo_profile_img[:,:,2] = echo_profile_img_bgr[:,:,0]
    #         # #echo_profile_img = echo_profile_img.reshape(echo_profile_img.shape[1], echo_profile_img.shape[0], echo_profile_img.shape[2])
            
    #         # plt.imshow(echo_profile_img)
    #         # plt.savefig('/data3/hyunchul/asl/Headset_silentspeech1/dl_model/images/sessions1/%s_echo.png'%truth[3])
    #         # plt.cla()
    #         print("test:", profile_data_piece.shape, normalized_imu_data.shape, truth_data)
    #         # print(truth_data)
    #         data_pairs += [(profile_data_piece, static_profiles_data_piece, truth_data, normalized_imu_data)]
    #         #print(truth)

    if (1):
        args_path = data_path
        #'/data3/hyunchul/asl/Headset_silentspeech1/asl_data/new_data/p00_0/dataset'
        sessions_list = get_sessions(args_path)
        #print(sessions_list) 
                
        train_data = []
        # loaded_gt = []
        # data_pairs = []
        
        session = session_num
        if is_train ==True:
            print('Train_session: : ', session)
        else:
            print('Test_session: : ', session)

        loaded_gt1 = []
        data_pairs1 = []
        
        file_echo = args_path + "/" + session + "/" + 'acoustic/diff'
        file_imus = args_path + "/" + session + "/" + 'imu'
        file_gnds = args_path + "/" + session + "/" + 'gnd_truth.txt'
        file_echo_list = sorted([f for f in os.listdir(file_echo)])
        file_imus_list = sorted([f for f in os.listdir(file_imus)])
        
        with open(file_gnds, 'r', encoding='utf-8') as f:
            gt = f.read()

        gt = gt.split("\n")[:-1]
        #print(gt)
        
        if len(file_echo_list) == len(file_imus_list):
            #print(session, file_echo_list, file_imus_list)
            for i in range(0, len(file_echo_list)):
                # ground truth

                gnd = int(file_echo_list[i].split('.')[0].split('_')[2])
                thdd = gt[gnd].split(',')[3]
                if thdd != '<SINGLE SHAKE>':
                    #print(thdd)
                    truth = gt[gnd].split(',')[0]
                    loaded_gt1 += [gt[gnd].split(',')]
                    #print(gnd, truth)
                    # IMU
                    File_data = np.loadtxt(file_imus+"/"+file_imus_list[i], dtype=str, delimiter=" ") 
                    all_raw = np.array(File_data, dtype=float)
                    all_imu = np.array(File_data, dtype=float)[:, :3]
                    all_imu_time = np.array(File_data, dtype=float)[:, 3:]
                    # echo
                    profiles = np.load(file_echo+"/"+file_echo_list[i]).astype('float32')
  
                    profiles = profiles[20:, :]
                    #print("aaa", profiles.shape)
                    profiles1 = profiles.copy()
                    #profiles1 = profiles1[20:, :]
                    # print('static check')
                    profile_data_piece = profiles1
                    # print("a:", profile_data_piece.shape)
                    profile_data_piece.shape = 1, -1, profile_data_piece.shape[1]
                    # print("b:", profile_data_piece.shape)
                    profile_data_piece = profile_data_piece.swapaxes(1, 2)
                    profiles = profile_data_piece
                    #print('aaa', profiles.shape) (1, 1624, 80)
                    
                    #profiles = profiles.reshape(1, profiles.shape[1], profiles.shape[0])
                    psampled_time, upsampled_imu_data = upsample_imu_data(all_imu_time, all_imu, profiles.shape[1])
                    normalized_imu_data, means, stds = normalize_imu_data(upsampled_imu_data)

                    if (0):
                        fig, (imu, acc) = plt.subplots(2)
                        fig.set_size_inches(10, 6)
                        # imu.plot(np.arange(len(normalized_imu_data)), normalized_imu_data[:, 0])
                        # imu.plot(np.arange(len(normalized_imu_data)), normalized_imu_data[:, 1])
                        # imu.plot(np.arange(len(normalized_imu_data)), normalized_imu_data[:, 2])
                        # imu.plot(upsampled_imu_data)
                        imu.plot(np.arange(len(normalized_imu_data)), normalized_imu_data[:, 0])
                        imu.plot(np.arange(len(normalized_imu_data)), normalized_imu_data[:, 1])
                        imu.plot(np.arange(len(normalized_imu_data)), normalized_imu_data[:, 2])
                        imu.margins(x = 0)
                        imu.set_ylim([-10, 10])
                        # n_channels=2
                        # coi=[0]
                        # tx_file=["fmcw20000_b4000_l100.wav"]
                        # signal='FMCW'
                        # if len(tx_file) != 0:
                        #     tx_files = []
                        #     if isinstance(tx_file, str):
                        #         tx_files = [tx_file]
                        #     else:
                        #         tx_files = tx_file
                        # elif signal.lower() == 'gsm':
                        #     tx_files = ['gsm.wav']
                        # elif signal.lower() == 'chirp':
                        #     tx_files = ['chirp.wav']
                        # elif signal.lower() == 'fmcw':
                        #     tx_files = ['fmcw.wav']

                        # SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
                        
                        # tx_signals = []
                        # for f in tx_files:
                        #     _, this_tx = load_audio(os.path.join(SCRIPTS_DIR, 'assets', f))
                        #     frame_length = this_tx.shape[0]
                        #     assert(frame_length == frame_len)    # frame_length must match
                        #     tx_signals += [this_tx]

                        # n_tx = len(tx_signals)
                        # if len(coi) == 0:
                        #     coi = list(range(n_channels))
                        # n_coi = len(coi)

                        diff_profiles_img = plot_profiles_split_channels(profiles1, 1, 20000000, -20000000)
                        #profiles11 = plot_profiles(profiles1, 20000000, -20000000)
                        acous_npy_img = cv2.cvtColor(np.float32(diff_profiles_img), cv2.COLOR_BGR2RGB)
                        acc.imshow(acous_npy_img.astype(np.uint16), aspect = 'auto')
        
                        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
                        # ax1.plot(upsampled_imu_data)
                        # profile_data_piece = profiles                           
                        # img = profile_data_piece.reshape(profile_data_piece.shape[1], profile_data_piece.shape[2])
                        # echo_profile_img_bgr = plot_profiles(img.T, 10000000, -10000000)
                        # echo_profile_img = echo_profile_img_bgr.copy()
                        # echo_profile_img[:,:,0] = echo_profile_img_bgr[:,:,2]
                        # echo_profile_img[:,:,2] = echo_profile_img_bgr[:,:,0]
                        
                        # ax2.imshow(echo_profile_img,aspect="auto")
                        # # ax2.set_title(str(float(truth[2]) - float(truth[1])))
                        # ax2.set_xticks([])
                        # ax2.set_yticks([])
                        # ax2.set_xlabel(data_file)
                        # #ax2.axhline(y=50, color='black', linestyle='-')
                        fig.savefig('./image_test/no_normal/11113_%s.png'%gt[gnd].split(',')[3])
                        print('save')
                        #fig.savefig(path + '/%s_%d.png'%(truth[3],cnt))
                        fig.clear()

                    #print("inputs:", profiles.shape, normalized_imu_data.shape, truth )
                    #data_pairs += [(profiles, profiles, truth, normalized_imu_data)]
                    if profiles.shape[1] > 280:
                        #data_pairs1 += [(profiles, profiles, truth, upsampled_imu_data)]
                        data_pairs1 += [(profiles, profiles, truth, normalized_imu_data)]
                    else: 
                        print("check error: ", file_echo_list[i])

    # if n_bad:
    #     print_and_log('%d bad data pieces' % n_bad)
    # print(np.mean(actual_lens))
    #return data_pairs, loaded_gt
    return data_pairs1, loaded_gt1

def generate_data1(input_config, data_config, is_train, data_path):
    '''
     ------------- You should code here to load the data from disk to the program-------------------------
     you should read the training data into the 'train_data', a list:
     step1: read image: img
     step2: save the img and ground-truth in the list 'train_data += [(img, ground-truth)]'

    '''

    # train_data = pickle.load(open('tmp_train_data.pkl', 'rb'))
    # test_data = pickle.load(open('tmp_test_data.pkl', 'rb'))
    # test_loaded_gt = pickle.load(open('tmp_test_loaded_gt.pkl', 'rb'))
    
    
    train_data = []
    test_data = []

    if is_train:
        print_and_log('Loading training data...')
        for p in data_config['train_sessions']:
            # data_file = os.path.join(data_config['root_folder'], p, data_config['data_file'])
            # static_file = os.path.join(data_config['root_folder'], p, data_config['static_file'])
            # truth_file = os.path.join(data_config['root_folder'], p, data_config['truth_file'])
            # config_file = os.path.join(data_config['root_folder'], p, data_config['config_file'])
            print_and_log('Loading from %s' % p)
            this_train_data, _ = read_from_folder1(p, data_path, is_train=True)
            train_data += this_train_data

    print_and_log('Loading testing data...')
    test_loaded_gt = []
    last_index_offset = 0   # avoid confusing the sliding window truths from multiple files
    for p in data_config['test_sessions']:
        # data_file = os.path.join(data_config['root_folder'], p, data_config['data_file'])
        # static_file = os.path.join(data_config['root_folder'], p, data_config['static_file'])
        # truth_file = os.path.join(data_config['root_folder'], p, data_config['truth_file'])
        # config_file = os.path.join(data_config['root_folder'], p, data_config['config_file'])
        print_and_log('Loading from %s' % p)
        if input_config['test_sliding_window']['applied']:
            last_index_offset = max([x[2][1] + input_config['test_target_length'] for x in test_data] + [0])
        this_test_data, this_loaded_gt = read_from_folder1(p, data_path, is_train=False)
        test_data += this_test_data
        test_loaded_gt += this_loaded_gt
    
    #print(test_loaded_gt)
    # '''

    # input_config['n_channels_loaded'] = 4
    # input_config['n_channels_prepared'] = 4
    # input_config['h_shift_frames'] = 0
    # input_config['channels_of_interest'] = list(range(4))

    if input_config['train_target_length'] <= 0:
        input_config['train_target_length'] = max([x[0].shape[1] for x in train_data] + [0])
    if input_config['test_target_length'] <= 0:
        input_config['test_target_length'] = max([x[0].shape[1] for x in test_data] + [0])
    
    print_and_log('Train/test target length: %d/%d frames' % (input_config['train_target_length'], input_config['test_target_length']))

    print("lens- tes:t", len(test_data))
    print("test_data[0][0].shape: ", test_data[0][0].shape)
    input_config['model_input_channels'] = test_data[0][0].shape[0]

    # train_data = train_data[:len(train_data) - len(train_data) % input_config['batch_size']]

    rand_indices = list(range(len(train_data)))
    random.shuffle(rand_indices)
    train_data_w_lengths = [(x[0].shape[1], rand_indices[i], x) for i, x in enumerate(train_data)]
    train_data_w_lengths.sort()
    train_data = [x[2] for x in train_data_w_lengths]
    

    # tmp, save the loading process
    # pickle.dump(train_data, open('tmp_train_data.pkl', 'wb'))
    # pickle.dump(test_data, open('tmp_test_data.pkl', 'wb'))
    # pickle.dump(test_loaded_gt, open('tmp_test_loaded_gt.pkl', 'wb'))
    data_splitter = DataSplitter(train_data, test_data, input_config['batch_size'], input_config['num_workers'], data_config['shuffle'], input_config)
    return data_splitter.train_loader, data_splitter.test_loader, test_loaded_gt


# control the number of threads (to prevent blocking ...)
# should work for both numpy / opencv
# it really depends on the CPU / GPU ratio ...
TARGET_NUM_THREADS = '4'
os.environ['OMP_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['MKL_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = TARGET_NUM_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = TARGET_NUM_THREADS
# os.environ['CUDA_VISIBLE_DEVICES'] = GPUs.select()
# numpy imports
# torch imports

# for visualization
# from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# the arg parser
parser = argparse.ArgumentParser(description='Hand pose from mutliple views')
parser.add_argument('--print-freq', default=30, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--valid-freq', default=3, type=int,
                    help='validation frequency (default: 5)')
parser.add_argument('-o', '--output', default='temp', type=str,
                    help='the name of output file')
parser.add_argument('-i', '--input', default='', type=str,
                    help='overwrites dataset.data_file')
parser.add_argument('--stacking', default='', choices=['vertical', 'channel', ''], type=str,
                    help='overwrites input.stacking')
parser.add_argument('-p', '--path', default='', type=str,
                    help='path to dataset parent folder, overwrites dataset.path')
parser.add_argument('-ts', '--test-sessions', default='', type=str,
                    help='overwrites dataset.test_sessions, comma separated, e.g. 5,6,7')
parser.add_argument('--exclude-sessions', default='', type=str,
                    help='remove these sessions from training AND testing, comma separated, e.g. 5,6,7')
parser.add_argument('--train-sessions', default='', type=str,
                    help='overwrites dataset.train_sessions, default using all but testing sessions for training, comma separated, e.g. 5,6,7')
parser.add_argument('-g', '--visible-gpu', default='', type=str,
                    help='visible gpus, comma separated, e.g. 0,1,2 (overwrites config file)')
# parser.add_argument('-m', '--mode', default='', type=str,
#                     help='the mode of training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=-1, type=int,
                    help='total epochs to run, overwrites optimizer.epochs')
parser.add_argument('--epochs-to-run', default=-1, type=int,
                    help='how many epochs left to run, overwrites --epochs')
parser.add_argument('--lr', default=0, type=float,
                    help='learning rate, overwrites optimizer.learning_rate')
parser.add_argument('--bb', default='',
                    help='backbone, overwrites network.backbone')
parser.add_argument('--bn', default=0, type=int,
                    help='batchsize, overwrites input.batch_size')
parser.add_argument('--coi', default='',
                    help='channels of interest, comma-separated, overwrites input.channels_of_interest')
parser.add_argument('-v', '--variance-files', action='append',
                    help='variance files to be added during training, overwrites input.variance_files')
parser.add_argument('--test', action='store_true',
                    help='test only')

parser.add_argument('-dp', '--datapath', type=str,
                    help='data_path')
# parser.add_argument('-a', '--augment', default=0, type=int,
#                     help='if use the image augment')
# parser.add_argument('--all', action='store_true',
#                     help='use the model to run over training and testing set')
# parser.add_argument('--train_file', nargs='+')

# parser.add_argument('--test_file', nargs='+')

# main function for training and testing

def save_array(pred, loaded_gt, filename, cm):
    if pred is not None:
        save_arr = [(loaded_gt[i][0], loaded_gt[i][1], loaded_gt[i][2], loaded_gt[i][3], pred[i]) for i in range(len(pred))]
        if False and cm:
            truths = [int(x[0]) for x in save_arr]
            preds = [x[4] for x in save_arr]
            labels = extract_labels(loaded_gt)
            generate_cm(np.array(truths), np.array(preds), labels, filename[:-4] + '_cm.png')
    else:
        save_arr = loaded_gt
    save_gt(save_arr, filename)

def main(args):
    # ===================== initialization for training ================================
    print(time.strftime('begin: %Y-%m-%d %H:%M:%S', time.localtime()))
    torch.cuda.empty_cache()
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)
    # parse args
    # best_metric = 100000.0
    best_metric = None
    metric_text = 'metric'
    args.start_epoch = 0
    data_path = args.datapath
    print('path:',data_path )
    
    torch.set_num_threads(int(TARGET_NUM_THREADS))

    config = load_config()  # load the configuration
    #print('Current configurations:')
    # pprint(config)
    #raise KeyboardInterrupt
    # prepare for output folder
    output_dir = args.output
    os.environ['CUDA_VISIBLE_DEVICES'] = config['network']['visible_devices']
    if len(args.visible_gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpu
    print_and_log('Using GPU # %s' % os.environ['CUDA_VISIBLE_DEVICES'])
    if len(args.path):
        config['dataset']['path'] = args.path.rstrip('/')
        config['dataset']['root_folder'] = os.path.join(config['dataset']['path'], 'dataset')
        output_dir = os.path.basename(config['dataset']['path'])
    if not args.output == 'temp':
        output_dir = (os.path.basename(config['dataset']['path']) + '_' + args.output).replace('__', '_')
    if len(args.input):
        config['dataset']['data_file'] = args.input
    if len(args.stacking):
        config['input']['stacking'] = args.stacking
    if len(args.test_sessions):
        all_session_names = os.listdir(config['dataset']['root_folder'])
        all_session_names.sort()
        test_sessions = [s for s in args.test_sessions.split(',') if len(s)]
        train_sessions = [s for s in args.train_sessions.split(',') if len(s)]
        exclude_sessions = [s for s in args.exclude_sessions.split(',') if len(s)]
        config['dataset']['train_sessions'] = []
        config['dataset']['test_sessions'] = []
        for ss in all_session_names:
            if not args.test and re.match(r'session_\w+', ss) is None:
                continue
            session_suffix = re.findall(r'session_(\w+)', ss)[0]
            if session_suffix in test_sessions:# and session_suffix not in exclude_sessions:
                config['dataset']['test_sessions'] += [ss]
            elif (len(args.train_sessions) == 0 or session_suffix in train_sessions) and session_suffix not in exclude_sessions:
                config['dataset']['train_sessions'] += [ss]
    if args.epochs > 0:
        config['optimizer']['epochs'] = args.epochs
    if args.epochs_to_run > 0:
        config['optimizer']['epochs'] = args.start_epoch + args.epochs_to_run
    if args.lr > 0:
        config['optimizer']['learning_rate'] = args.lr
    if args.bn > 0:
        config['input']['batch_size'] = args.bn
    if len(args.bb) > 0:
        config['network']['backbone'] = args.bb
    if len(args.coi) > 0:
        config['input']['channels_of_interest'] = [int(x) for x in args.coi.split(',')]
    config['input']['variance_files'] = args.variance_files
    torch.cuda.empty_cache()
    ckpt_folder = os.path.join('./ckpt/%s'%output_dir.split('_')[0], output_dir)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    log_path = os.path.join(ckpt_folder, 'logs.txt')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info(time.strftime('begin: %Y-%m-%d %H:%M:%S', time.localtime()))
    # use spawn for mp, this will fix a deadlock by OpenCV (do not need)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    # fix the random seeds (the best we can)
    fixed_random_seed = 20220217
    torch.manual_seed(fixed_random_seed)
    np.random.seed(fixed_random_seed)
    random.seed(fixed_random_seed)

    #print("config:", config)
    # set up transforms and dataset
    # ===================== packet the training data and testinf data ================================
    train_dataset, val_dataset, test_gt = generate_data1(input_config=config['input'], data_config=config['dataset'], is_train=(not args.test), data_path = data_path)

    if config['network']['loss_type'] == 'ce':  # classification task, determine output dimension based on number of labels
        #config['network']['output_dims'] = max([int(x[0]) for x in test_gt]) + 1
        config['network']['output_dims'] = 27
        print_and_log('Classification task detected, output dimension: %d' % config['network']['output_dims'])
    elif config['network']['loss_type'] == 'ctc':  # classification task, determine output dimension based on number of labels
        #config['network']['output_dims'] = max([max([int(xx) for xx in x[0].split()]) for x in test_gt]) + 2
        config['network']['output_dims'] = 27
        print_and_log('Classification task detected, output dimension: %d' % config['network']['output_dims'])
    else:
        config['network']['output_dims'] = test_gt[0].shape[0] - 1
        print_and_log('Regression task detected, output dimension: %d' % config['network']['output_dims'])
    config['network']['input_channels'] = config['input']['model_input_channels']
    print( "input_channel", config['input']['model_input_channels'])
    print_and_log(time.strftime('finish data: %Y-%m-%d %H:%M:%S', time.localtime()))
    # create model w. loss
    
    modelA = ModelBuilder(config['network'])  # load the designed model
    modelB = ModelBuilder(config['network'])  # load the designed model
    #modelA.decoder.fc1 = torch.nn.Linear(512, 256)
    master_gpu = config['network']['devices'][0]
    class MyEnsemble(nn.Module):
        def __init__(self, modelA, modelB, targets):
            super(MyEnsemble, self).__init__()
            self.modelA_encoder = modelA.encoder
            self.modelB_encoder = modelB.encoder
            self.targets = targets
            #self.modelA_decoder = modelB.decoder
            # self.avgpool= nn.AdaptiveAvgPool2d((512, 1))
            # self.dropout = nn.Dropout(p=0.01, inplace=False)
            # self.fc1 = nn.Linear(512, 27, bias=True)
            self.avgpool = nn.AdaptiveAvgPool2d((512, 1))
            self.dropout = nn.Dropout(p=0.01, inplace=False)
            # print('in_channels = ', self.in_channels)
            # print('input_views = ', self.input_views)
            # self.fc = nn.Linear(self.in_channels, self.output_dims, bias=True)
            self.fc1 = nn.Linear(1024, 27, bias=True)

            # self.fc1 = nn.Linear(1024, 500, bias=True)
            # self.fc2 = nn.Linear(500, 250, bias=True)
            # self.fc3 = nn.Linear(250, 27, bias=True)
            #self.reset_params()
        
        # def reset_params(self):
        #     # manuall init fc params
        #     nn.init.normal_(self.fc1.weight, 0.0, 0.02)
        #     nn.init.normal_(self.fc2.weight, 0.0, 0.02)
        #     nn.init.normal_(self.fc3.weight, 0.0, 0.02)
        #     self.fc_bias = None
        #     #nn.init.normal_(self.fc2.weight, 0.0, self.fc_std)
        #     if self.fc_bias is None:
        #         nn.init.constant_(self.fc1.bias, 0.0)
        #         nn.init.constant_(self.fc2.bias, 0.0)
        #         nn.init.constant_(self.fc3.bias, 0.0)
        #         #nn.init.constant_(self.fc2.bias, 0.0)
        #     else:
        #         self.fc1.bias.data = torch.from_numpy(self.fc_bias.copy())
        #         self.fc2.bias.data = torch.from_numpy(self.fc_bias.copy())
        #         self.fc3.bias.data = torch.from_numpy(self.fc_bias.copy())
        #         #self.fc2.bias.data = torch.from_numpy(self.fc_bias.copy())

            # self.fc2 = nn.Linear(128, 27)
            
        def forward(self, x1, x2, targets):
            x1 = self.modelA_encoder(x1)
            x2 = self.modelB_encoder(x2)
            #print(x1[0].shape,x1[1].shape, x1[2].shape, x1[3].shape,len(x1))
            in_vec1 = x1[-1]
            in_vec2 = x2[-1]
#---------          
            #print(in_vec1.shape, in_vec2.shape)

            in_vec1 = in_vec1.transpose(1, 2)         # feats: N x C x L x height -> N x L x C x h
            #print(in_vec1.shape)
            in_vec1 = self.avgpool(in_vec1)  
            #print(in_vec1.shape)         # n x l x c x 1
            n, c, l, h = in_vec1.shape
            
            in_vec1 = in_vec1.reshape(n, c, l)
            #print(in_vec1.shape)
            in_vec1 -= in_vec1.min(2, keepdim=True)[0]
            in_vec1 /= in_vec1.max(2, keepdim=True)[0]  

            in_vec1 = in_vec1.reshape(n, l, c)  
            #print(in_vec1.shape)

            # in_vec2 = in_vec2.expand(-1, -1, -1, in_vec1.size(3))
            # print(in_vec1.shape, in_vec2.shape)
            
            in_vec2= in_vec2.transpose(1, 2)         # feats: N x C x L x height -> N x L x C x h
            #print(in_vec2.shape)
            in_vec2 = self.avgpool(in_vec2)  
            #print(in_vec2.shape)         # n x l x c x 1
            n, c, l, h = in_vec2.shape

            in_vec2 = in_vec2.reshape(n, c, l)
            #print(in_vec2.shape)
            in_vec2 -= in_vec2.min(2, keepdim=True)[0]
            in_vec2 /= in_vec2.max(2, keepdim=True)[0]  

            in_vec2 = in_vec2.reshape(n, l, c)  
            #print(in_vec2.shape)

            # norm

            
            
            in_vec = torch.cat((in_vec1, in_vec2), dim=1)
            #print(in_vec.shape)
            n, c, l = in_vec.shape
            in_vec = in_vec.reshape(n, l, c)  

            if (0):
                # vec2 --> imu
                in_vec = in_vec2
                n, c, l = in_vec.shape
                in_vec = in_vec.reshape(n, l, c)  
                print(in_vec.shape)
            # n, c, l, h = in_vec.shape
            # in_vec = in_vec.transpose(1, 2)         # feats: N x C x L x height -> N x L x C x h
            # print(in_vec.shape)
            # in_vec = self.avgpool(in_vec)  
            # print(in_vec.shape)         # n x l x c x 1
            # in_vec = in_vec.reshape(n, l, c)  
            # print(in_vec.shape)


#--------- ver1 
            # print(in_vec1.shape, in_vec2.shape)
            # in_vec2 = in_vec2.expand(-1, -1, -1, in_vec1.size(3))
            # print(in_vec1.shape, in_vec2.shape)
            # in_vec = torch.cat((in_vec1, in_vec2), dim=1)
            # print(in_vec.shape)
            # n, c, l, h = in_vec.shape
            # in_vec = in_vec.transpose(1, 2)         # feats: N x C x L x height -> N x L x C x h
            # print(in_vec.shape)
            # in_vec = self.avgpool(in_vec)  
            # print(in_vec.shape)         # n x l x c x 1
            # in_vec = in_vec.reshape(n, l, c)  
            # print(in_vec.shape)
#---------
  
            # #print(in_vec.shape)
            # in_vec1 = in_vec1.transpose(1, 2)         # feats: N x C x L x height -> N x L x C x h
            # in_vec1 = self.avgpool(in_vec1)           # n x l x c x 1
            # in_vec1 = in_vec1.reshape(n, l, c)  
            # #in_vec1 = F.normalize(in_vec1, p=2, dim=2)
        
            # x2 = self.modelB_encoder(x2)
            # #print(x1[0].shape,x1[1].shape, x1[2].shape, x1[3].shape,len(x1))
            # in_vec2 = x2[-1]
            # n, c, l, h = in_vec2.shape
            # #print(in_vec.shape)
            # in_vec2 = in_vec2.transpose(1, 2)         # feats: N x C x L x height -> N x L x C x h
            # in_vec2 = self.avgpool(in_vec2)           # n x l x c x 1
            # in_vec2 = in_vec2.reshape(n, l, c)  
            # #in_vec2 = F.normalize(in_vec2, p=2, dim=2)
            
            # #print("both", in_vec1.shape, in_vec2.shape)
            
            # in_vec = torch.cat((in_vec1, in_vec2), dim=2)
            # # #print("con",in_vec.shape)


            #out = torch.stack([F.log_softmax(self.dropout( F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(in_vec[i])))))) ), dim=-1) for i in range(n)])
            out = torch.stack([F.log_softmax(self.dropout(  self.fc1(in_vec[i])), dim=-1) for i in range(n)])
            #out = torch.stack([F.log_softmax(self.dropout(  F.relu(self.fc1(in_vec[i]))), dim=-1) for i in range(n)])

            #out1 = torch.stack([F.log_softmax(self.dropout(  self.fc1(in_vec2[i])), dim=-1) for i in range(n)])

            #print(out.shape)
            #x1 = self.modelA_decoder(x1)
            #print(x1.shape)
          

            criterion = nn.CTCLoss(blank=out.size(2) - 1)    
            n = out.size(0) # n - batch size
            target_lengths = torch.IntTensor([len(x.split()) for x in targets]).cuda(
                master_gpu, non_blocking=True) # network_config = 1??
            
            targets_cuda = []
            
            for t in targets:
                targets_cuda += [int(x) for x in t.split()]
            targets_cuda = torch.IntTensor(targets_cuda).cuda(
                master_gpu, non_blocking=True)
            
            #outputs = x1.log_softmax(2).detach().requires_grad_()
            outputs = out
            #print(outputs.shape, x1.size(2))
            pred_lengths = torch.IntTensor(n).fill_(out.shape[1])
            loss = criterion(outputs.transpose(0, 1), targets_cuda, pred_lengths, target_lengths)
            #loss = criterion(outputs.transpose(0, 1), targets_cuda, pred_lengths, target_lengths)
            #loss1 = criterion(out.transpose(0, 1), targets_cuda, pred_lengths, target_lengths)
            #loss2 = criterion(out1.transpose(0, 1), targets_cuda, pred_lengths, target_lengths)
            #loss = (loss1 + loss2)/2.0
            return outputs, loss
        
    # class MyEnsemble(nn.Module):
    #     def __init__(self, modelA, modelB, targets):
    #         super(MyEnsemble, self).__init__()
    #         self.modelA_encoder = modelA.encoder
    #         self.modelB_encoder = modelB.encoder
    #         self.targets = targets
    #         #self.modelA_decoder = modelB.decoder
    #         # self.avgpool= nn.AdaptiveAvgPool2d((512, 1))
    #         # self.dropout = nn.Dropout(p=0.01, inplace=False)
    #         # self.fc1 = nn.Linear(512, 27, bias=True)
    #         self.avgpool = nn.AdaptiveAvgPool2d((512, 1))
    #         self.dropout = nn.Dropout(p=0.01, inplace=False)
    #         # print('in_channels = ', self.in_channels)
    #         # print('input_views = ', self.input_views)
    #         # self.fc = nn.Linear(self.in_channels, self.output_dims, bias=True)
    #         self.fc1 = nn.Linear(1024, 27, bias=True)

    #         # self.fc1 = nn.Linear(1024, 500, bias=True)
    #         # self.fc2 = nn.Linear(500, 250, bias=True)
    #         # self.fc3 = nn.Linear(250, 27, bias=True)
    #         #self.reset_params()
        
    #     # def reset_params(self):
    #     #     # manuall init fc params
    #     #     nn.init.normal_(self.fc1.weight, 0.0, 0.02)
    #     #     nn.init.normal_(self.fc2.weight, 0.0, 0.02)
    #     #     nn.init.normal_(self.fc3.weight, 0.0, 0.02)
    #     #     self.fc_bias = None
    #     #     #nn.init.normal_(self.fc2.weight, 0.0, self.fc_std)
    #     #     if self.fc_bias is None:
    #     #         nn.init.constant_(self.fc1.bias, 0.0)
    #     #         nn.init.constant_(self.fc2.bias, 0.0)
    #     #         nn.init.constant_(self.fc3.bias, 0.0)
    #     #         #nn.init.constant_(self.fc2.bias, 0.0)
    #     #     else:
    #     #         self.fc1.bias.data = torch.from_numpy(self.fc_bias.copy())
    #     #         self.fc2.bias.data = torch.from_numpy(self.fc_bias.copy())
    #     #         self.fc3.bias.data = torch.from_numpy(self.fc_bias.copy())
    #     #         #self.fc2.bias.data = torch.from_numpy(self.fc_bias.copy())

    #         # self.fc2 = nn.Linear(128, 27)
            
    #     def forward(self, x1, x2, targets):
    #         x1 = self.modelA_encoder(x1)
    #         #print(x1[0].shape,x1[1].shape, x1[2].shape, x1[3].shape,len(x1))
    #         in_vec1 = x1[-1]
    #         n, c, l, h = in_vec1.shape
    #         #print(in_vec.shape)
    #         in_vec1 = in_vec1.transpose(1, 2)         # feats: N x C x L x height -> N x L x C x h
    #         in_vec1 = self.avgpool(in_vec1)           # n x l x c x 1
    #         in_vec1 = in_vec1.reshape(n, l, c)  
    #         in_vec1 = F.normalize(in_vec1, p=2, dim=2)
        
    #         x2 = self.modelA_encoder(x2)
    #         #print(x1[0].shape,x1[1].shape, x1[2].shape, x1[3].shape,len(x1))
    #         in_vec2 = x2[-1]
    #         n, c, l, h = in_vec2.shape
    #         #print(in_vec.shape)
    #         in_vec2 = in_vec2.transpose(1, 2)         # feats: N x C x L x height -> N x L x C x h
    #         in_vec2 = self.avgpool(in_vec2)           # n x l x c x 1
    #         in_vec2 = in_vec2.reshape(n, l, c)  
    #         in_vec2 = F.normalize(in_vec2, p=2, dim=2)
            
    #         #print("both", in_vec1.shape, in_vec2.shape)
            
    #         in_vec = torch.cat((in_vec1, in_vec2), dim=2)
    #         #print("con",in_vec.shape)


    #         #out = torch.stack([F.log_softmax(self.dropout( F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(in_vec[i])))))) ), dim=-1) for i in range(n)])
    #         out = torch.stack([F.log_softmax(self.dropout(  self.fc1(in_vec[i])), dim=-1) for i in range(n)])

    #         #print(out.shape)
    #         #x1 = self.modelA_decoder(x1)
    #         #print(x1.shape)
          

    #         criterion = nn.CTCLoss(blank=out.size(2) - 1)    
    #         n = out.size(0) # n - batch size
    #         target_lengths = torch.IntTensor([len(x.split()) for x in targets]).cuda(
    #             master_gpu, non_blocking=True) # network_config = 1??
            
    #         targets_cuda = []
            
    #         for t in targets:
    #             targets_cuda += [int(x) for x in t.split()]
    #         targets_cuda = torch.IntTensor(targets_cuda).cuda(
    #             master_gpu, non_blocking=True)
            
    #         #outputs = x1.log_softmax(2).detach().requires_grad_()
    #         outputs = out
    #         #print(outputs.shape, x1.size(2))
    #         pred_lengths = torch.IntTensor(n).fill_(out.shape[1])
    #         loss = criterion(outputs.transpose(0, 1), targets_cuda, pred_lengths, target_lengths)

    #         return outputs, loss
 
    model = MyEnsemble(modelA, modelB, targets=None)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    model = model.cuda(master_gpu)  # load model from CPU to GPU

########################    
    # model = ModelBuilder(config['network'])  # load the designed model
    # # GPU you will use in training
    # master_gpu = config['network']['devices'][0]
    # model = model.cuda(master_gpu)  # load model from CPU to GPU
    # # create optimizer
    # optimizer = create_optim(model, config['optimizer'])  # gradient descent
###########################
    # data parallel
    # if you want use multiple GPU
    model = nn.DataParallel(model, device_ids=config['network']['devices'])
    logging.info(model)
    # set up learning rate scheduler
    if not args.test:
        num_iters_per_epoch = len(train_dataset)
        scheduler = create_scheduler(
            optimizer, config['optimizer']['schedule'],
            config['optimizer']['epochs'], num_iters_per_epoch)

    # ============================= retrain the trained model (if need usually not) =========================================
    # resume from a checkpoint?
    if args.resume:
        #not args.test = 0
        print_and_log('loading trained model.....')
        if os.path.isfile(args.resume):
            print_and_log('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(master_gpu))
            # args.start_epoch = 0
            args.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_metric']
            # encoder_only_dict = {x: checkpoint['state_dict'][x] for x in checkpoint['state_dict'] if 'decoder' not in x}
            # encoder_only_dict.update({x: model.state_dict()[x] for x in model.state_dict() if 'decoder' in x})
            # model.load_state_dict(encoder_only_dict)
            model.load_state_dict(checkpoint['state_dict'])
            if args.epochs_to_run > 0:
                config['optimizer']['epochs'] = args.start_epoch + args.epochs_to_run
            # only load the optimizer if necessary
            if not args.test:
                # best_metric = ['best_metric']
                scheduler = create_scheduler(
                    optimizer, config['optimizer']['schedule'],
                    config['optimizer']['epochs'] - args.start_epoch, num_iters_per_epoch)
                # optimizer.load_state_dict(checkpoint['optimizer'])
                # scheduler.load_state_dict(checkpoint['scheduler'])
            print_and_log('=> loaded checkpoint {} (epoch {}, metric {:.3f}, best_metric {:.3f})'
                  .format(args.resume, checkpoint['epoch'], checkpoint['ckpt_metric'], best_metric))
        else:
            print_and_log('=> no checkpoint found at {}'.format(args.resume))
            return

    # =================================== begin training =========================================
    # training: enable cudnn benchmark
    cudnn.enabled = True
    cudnn.benchmark = True

    # model architecture
    model_arch = '{:s}-{:s}'.format(
        config['network']['backbone'], config['network']['decoder'])

    # start the training
    if not args.test:
        # if not os.path.isfile(args.resume):
        if best_metric is None:
            if config['network']['loss_type'] == 'ce':
                best_metric = 0
            else:
                best_metric = np.inf
        # save the current config
        with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
            pprint(config, stream=fid)
        # print('Training model {:s} ...'.format(model_arch))
        #save_test_data(val_dataset, config_filename)
        torch.cuda.empty_cache()

        for epoch in range(args.start_epoch, config['optimizer']['epochs']):
            # train for one epoch
            # print('epoch', epoch)
            #print(time.strftime('begin: %Y-%m-%d %H:%M:%S', time.localtime()))
            # (training data, model, others (configuration, gradient descent))
            train(train_dataset, model, optimizer,
                  scheduler, epoch, args, config)
            # torch.cuda.empty_cache()

            # evaluate on validation set once in a while
            # test on every epoch at the end of training
            # Note this will also run after first epoch (make sure training is on track)
            if epoch % args.valid_freq == 0 \
                    or (epoch > 0.9 * config['optimizer']['epochs']):
                # run bn once before validation
                prec_bn(train_dataset, model, args, config)
                # (testing data, model, others (configuration, gradient descent))
                metric, loss, pred_array, fake_gts = validate(val_dataset, model, args, config)
                # if epoch // args.valid_freq == 0:
                # print(metric)
                # remember best metric and save checkpoint
                if config['network']['loss_type'] == 'ctc' and config['input']['test_sliding_window']['applied']:
                    # print(pred_array[:10])
                    metric, pred_array, _ = wer_sliding_window(pred_array, fake_gts, test_gt, config['input']['test_sliding_window']['pixels_per_label'])
                    # print(pred_array[:10])
                display_str = '**** testing loss: {:.4f}, metric: {:.4f}, '.format(loss, metric)
                if config['network']['loss_type'] == 'ce':
                    is_best = metric > best_metric
                    best_metric = max(metric, best_metric)
                else:
                    is_best = metric < best_metric
                    best_metric = min(metric, best_metric)
                if config['network']['loss_type'] in ['ce', 'ctc']:
                    save_array(pred_array, test_gt, os.path.join(ckpt_folder, 'ckpt_pred.txt'), config['network']['loss_type'] == 'ce')
                    exact_match_acc = np.mean([test_gt[x][0] == pred_array[x] for x in range(len(test_gt))])
                    display_str += 'exact match acc: %.2f%%, ' % (100 * exact_match_acc)
                else:
                    save_array(None, pred_array, os.path.join(ckpt_folder, 'ckpt_pred.npy'), False)
                    save_array(None, test_gt, os.path.join(ckpt_folder, 'test_gt.npy'), False)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_arch': model_arch,
                    'state_dict': model.state_dict(),
                    'ckpt_metric': metric,
                    'best_metric': best_metric,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, file_folder=ckpt_folder)
                # print('saving')
                if is_best:
                    display_str += f'\033[92mbest_%s = %.4f\033[0m' % (metric_text, best_metric)
                    if config['network']['loss_type'] in ['ce', 'ctc']:
                        save_array(pred_array, test_gt, os.path.join(ckpt_folder, 'best_pred.txt'), config['network']['loss_type'] == 'ce')
                    else:
                        save_array(None, pred_array, os.path.join(ckpt_folder, 'best_pred.npy'), False)
                        save_array(None, test_gt, os.path.join(ckpt_folder, 'test_gt.npy'), False)
                else:
                    display_str += f'best_%s = %.4f' % (metric_text, best_metric)
                display_str += ' ' * 20
                print_and_log(display_str)
                # manually reset mem
            torch.cuda.empty_cache()
################################ save the file ###########################################

    if args.test:
        metric_test, loss_test, pred_all_test, fake_gts = validate(val_dataset, model, args, config)
        if config['network']['loss_type'] == 'ctc' and config['input']['test_sliding_window']['applied']:
            # print(pred_all_test[:10])
            # print(fake_gts[:10])
            metric_test, pred_all_test, raw_preds_all = wer_sliding_window(pred_all_test, fake_gts, test_gt, config['input']['test_sliding_window']['pixels_per_label'])
            # print(raw_preds_all.shape)
            np.save('raw_preds_all.npy', raw_preds_all)
        print_and_log('**** Testing loss: %.4f, metric: %.4f ****' % (loss_test, metric_test))
        if config['network']['loss_type'] in ['ce', 'ctc']:
            save_array(pred_all_test, test_gt, os.path.join(ckpt_folder, 'test_pred.txt'), config['network']['loss_type'] == 'ce')
            # for i in range(len(test_gt)):
            #     print(test_gt[i][0], pred_all_test[i])
            #     if test_gt[i][0] != pred_all_test[i]:
            #         print(test_gt[i])
            all_acc = np.mean([test_gt[x][0] == pred_all_test[x] for x in range(len(pred_all_test))])
            print_and_log('Exact match acc: %.4f' % all_acc)
        else:
            save_array(None, pred_all_test, os.path.join(ckpt_folder, 'test_pred.npy'), False)
            save_array(None, test_gt, os.path.join(ckpt_folder, 'test_gt.npy'), False) 
        # save_array(pred_all_test, test_gt, os.path.join(ckpt_folder, 'test_pred.txt'))

    print_and_log(time.strftime('end: %Y-%m-%d %H:%M:%S', time.localtime()))

def train(train_loader, model, optimizer, scheduler, epoch, args, config=None):
    '''Training the model'''
    # set up meters
    num_iters = len(train_loader)
    batch_time = AverageMeter()
    # loss is our err here
    losses = AverageMeter()
    metrics = AverageMeter()

    # switch to train mode
    model.train()
    master_gpu = config['network']['devices'][0]
    end = time.time()

    loss_type = config['network']['loss_type']
    criterion = get_criterion(loss_type)

    output_str_length = 0
    num = 0
    for i, (input_arr_raw, target) in enumerate(train_loader):
        input_arr = input_arr_raw[0]
        #print(input_arr.shape, input_arr)
        input_imu = input_arr_raw[1]
        input_arr = input_arr.cuda(master_gpu, non_blocking=True)
        input_imu = input_imu.cuda(master_gpu, non_blocking=True)
        # print(target, type(target))
        #print("shape:", input_arr.shape, input_imu.shape)
        # input_arr = augment(input_arr, config['dataset'])
        #print(type(input_arr), input_arr.shape)
        #raise KeyboardInterrupt
        input_arr = input_arr.cuda(master_gpu, non_blocking=True)
        if loss_type == 'ce':
            target = torch.LongTensor([int(x) for x in target])
        if not loss_type == 'ctc':
            target = target.cuda(master_gpu, non_blocking=True)
       
       
       
        # compute output
        #output, loss = model(input_arr, targets=target)
        # if (1):
        #     print(input_arr.shape, input_imu.shape, target)
        #     torch.save(input_arr, './data_sample/tensor_arr_%d.pt'%num)
        #     torch.save(input_imu, './data_sample/tensor_imu_%d.pt'%num)
        #     np.save('./data_sample/target_array_%d.pt'%num, np.array(target))
        #     num+=1
        
        output, loss = model(input_arr, input_imu, target)


        loss = loss.mean()
        # input_arr.cpu()
        # output.cpu()
        # target.cpu()
        # compute gradient and do SGD step
        # !!!! important (update the parameter of your model with gradient descent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metric, preds, raw_preds = criterion(output, target)
        # if epoch % 5 == 0:
        #     print(target, preds)

        losses.update(loss.data.item(), input_arr.size(0))
        metrics.update(metric, input_arr.size(0))

        del loss
        del output

        # printing the loss of traning
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            # make sure tensors are properly detached from the graph

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            # printing
            output_str = 'Epoch: [{:3d}][{:4d}/{:4d}], Time: {:.2f} ({:.2f}), loss: {:.3f} ({:.3f}), metric: {:.4f} ({:.4f})'\
                            .format(epoch, i, num_iters, batch_time.val,
                                    batch_time.avg, losses.val,
                                    losses.avg, metrics.val, metrics.avg)
            print_and_log(output_str, end='\r')
            output_str_length = max(output_str_length, len(output_str))
        # step the lr scheduler after each iteration
        scheduler.step()

    # print the learning rate
    lr = scheduler.get_last_lr()[0]
    output_str = 'Epoch {:d} finished with lr={:.6f}, loss={:.3f}, metric={:.4f}'.format(epoch, lr, losses.avg, metrics.avg)
    output_str += ' ' * (output_str_length - len(output_str) + 1)
    print_and_log(output_str)
    # lr = scheduler.get_lr()[0]
    # print('\nEpoch {:d} finished with lr={:f}'.format(epoch + 1, lr))
    # log metric
    # writer.add_scalars('data/metric', {'train' : metric.avg}, epoch + 1)i
    # print(metric.avg)
    return metrics.avg


def validate(val_loader, model, args, config):
    '''Test the model on the validation set'''
    # set up meters
    batch_time = AverageMeter()
    metrics = AverageMeter()
    losses = AverageMeter()

    # metric_action = AverageMeter()
    # metric_peak = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # master_gpu = config['network']['devices'][0]
    end = time.time()

    # prepare for outputs
    pred_list = []
    # truth_list = []

    loss_type = config['network']['loss_type']
    criterion = get_criterion(loss_type)
    # criterion = get_criterion(loss_type)
    # criterion = Point_dis_loss
    # criterion = calculate_dis
    # criterion = nn.functional.l1_loss
    # criterion = nn.functional.cross_entropy

    output_str_length = 0
    sliding_window = (loss_type == 'ctc' and config['input']['test_sliding_window']['applied'])
    # if sliding_window:
    fake_gts = []
        # raw_preds_all = []
    master_gpu = config['network']['devices'][0]
    # loop over validation set
    for i, (input_arr_raw, target) in enumerate(val_loader):
        input_arr = input_arr_raw[0]
        input_imu = input_arr_raw[1]
        input_arr = input_arr.cuda(master_gpu, non_blocking=True)
        input_imu = input_imu.cuda(master_gpu, non_blocking=True)
        # print(target)
        # print(input_arr.shape)
        if loss_type == 'ce':
            target = torch.LongTensor([int(x) for x in target])
        if not loss_type == 'ctc':
            target = target.cuda(config['network']['devices'][0], non_blocking=True)
        if sliding_window:
            for s, e in zip(target[0].numpy(), target[1].numpy()):
                fake_gts += [(s, e)]
            target = None
        # forward the model
        # print(input_arr.shape)
        with torch.no_grad():
            #output, loss = model(input_arr, targets=target)
            #output, loss = model(input_arr, target)
            output, loss = model(input_arr, input_imu, target)
        # loss = loss.mean()
        # print(type(output.data), output.data.shape)

        # measure metric and record loss
        metric, pred, raw_preds = criterion(output, target, require_pred=True)
        # print(pred)
        # print(raw_preds)
        # # err = nn.functional.l1_loss(pred, truth)
        # err = criterion(output, target)
        if loss is not None:
            losses.update(loss.data.item(), input_arr.size(0))
        if metric is not None:
            # print(metric)
            metrics.update(metric, input_arr.size(0))
        # if sliding_window:
        #     raw_preds_all += raw_preds

       # print(type(pred), pred.shape)
        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # append the output list
        if sliding_window:
            pred_list.append(raw_preds)
        else:
            pred_list.append(pred)

        # printing
        if i % (args.print_freq * 2) == 0 or i == len(val_loader) - 1:
            output_str = 'Test: [{:4d}/{:4d}], Time: {:.2f} ({:.2f}), loss {:.2f} ({:.2f}), metric {:.4f} ({:.4f})'.format(i + 1, len(val_loader),
                                                  batch_time.val, batch_time.avg,
                                                  losses.val, losses.avg,
                                                  metrics.val, metrics.avg)
            print_and_log(output_str, end='\r')
            output_str_length = max(output_str_length, len(output_str))
            # print('Test: [{:d}/{:d}]\t'
            #       'Time {:.2f} ({:.2f})\t'
            #       'MSD {:.3f} ({:.3f})\t'.format(
            #           i, len(val_loader), batch_time.val, batch_time.avg, metric.val, metric.avg), end='\r')
    # output_str = '**** testing loss: {:.4f}, metric: {:.4f}'.format(losses.avg, metrics.avg)
    # output_str += ' ' * (output_str_length - len(output_str) + 1)
    # print_and_log(output_str, end=' ') 
    # print('\n******MSD {:3f}'.format(metric.avg))
    pred_list = np.concatenate(pred_list)
    # print(pred_list[:100])
    return metrics.avg, losses.avg, pred_list, fake_gts

def prec_bn(train_loader, model, args, config):
    '''Aggreagate precise BN stats on train set'''
    # set up meters
    batch_time = AverageMeter()
    # switch to train mode (but do not require gradient / update)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.05
    end = time.time()
    master_gpu = config['network']['devices'][0]

    # loop over validation set
    for i, (input_arr_raw, target) in enumerate(train_loader):
        # forward the model
        input_arr = input_arr_raw[0]
        input_imu = input_arr_raw[1]
        input_arr = input_arr.cuda(master_gpu, non_blocking=True)
        input_imu = input_imu.cuda(master_gpu, non_blocking=True)
        with torch.no_grad():
            #_ = model(input_arr)
           # _,_ = model(input_arr, target)
            _,_ = model(input_arr,input_imu, target)

        # printing
        if i % (args.print_freq * 2) == 0:
            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / (args.print_freq * 2))
            end = time.time()
            print_and_log('Prec BN: [{:d}/{:d}], Time: {:.2f} ({:.2f})'.format(
                      i, len(train_loader), batch_time.val, batch_time.avg), end='\r')
    # print('\n')
    return


################################################################################
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)