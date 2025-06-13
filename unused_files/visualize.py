import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import math

def plot_profiles(profiles, max_val=None, min_val=None):
    max_h = 0       # red
    min_h = 120     # blue
    if not max_val:
        max_val = np.max(profiles)
    if not min_val:
        min_val = np.min(profiles)
    #print(max_val, min_val)
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

#change from visualizing as 1 channel to 4 channels
def vis(input):
    img_input = input.copy()
    diff_profiles_img = plot_profiles_split_channels(img_input.T, 1, 50000000, -50000000)
    #profiles11 = plot_profiles(profiles1, 20000000, -20000000)
    acous_npy_img = cv2.cvtColor(np.float32(diff_profiles_img), cv2.COLOR_BGR2RGB)
    plt.imshow(acous_npy_img.astype(np.uint16), aspect = 'auto')
    plt.savefig('./fake_img.png')

def vis_save(input):
    img_input = input.copy()
    diff_profiles_img = plot_profiles_split_channels(img_input.T, 1, 50000000, -50000000)
    #profiles11 = plot_profiles(profiles1, 20000000, -20000000)
    acous_npy_img = cv2.cvtColor(np.float32(diff_profiles_img), cv2.COLOR_BGR2RGB)
    plt.imshow(acous_npy_img.astype(np.uint16), aspect = 'auto')
    plt.savefig('./fake_img.png')

def vis_out(input):
    img_input = input.copy()
    diff_profiles_img = plot_profiles_split_channels(img_input.T, 1, 50000000, -50000000)
    #profiles11 = plot_profiles(profiles1, 20000000, -20000000)
    acous_npy_img = cv2.cvtColor(np.float32(diff_profiles_img), cv2.COLOR_BGR2RGB)
    return acous_npy_img.astype(np.uint16)

def vis_out_one_channel(input):
    img_input = input.copy()
    diff_profiles_img = plot_profiles(img_input, 50000000, -50000000)
    # diff_profiles_img = plot_profiles_split_channels(img_input.T, 1, 50000000, -50000000)
    #profiles11 = plot_profiles(profiles1, 20000000, -20000000)
    acous_npy_img = cv2.cvtColor(np.float32(diff_profiles_img), cv2.COLOR_BGR2RGB)
    return acous_npy_img.astype(np.uint16)

data_dir = 'glasses_train_8/diff'
npy_files = glob.glob(os.path.join(data_dir, '*.npy'))

# #test visualizing one npy file
# first_image = np.load(npy_files[0])
# vis(first_image[3])

first_image = np.load(npy_files[0])
channel_4 = first_image[3]

# Use the actual min/max for better contrast
img = plot_profiles(channel_4)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 4))
plt.imshow(img_rgb, aspect='auto')
plt.title("Fourth Channel")
# plt.axis('off')
plt.show()


# img = vis_out_one_channel((first_image[0]))
# plt.imshow(img, aspect = 'auto')
# plt.show()

# height, width = first_image[0].shape
# dpi = 100  # or any value you like, dots per inch
# figsize = (width / dpi, height / dpi)

# fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
# img = vis_out_one_channel(first_image[0])
# ax.imshow(img, aspect='auto')
# plt.show()


#visualize all the A files in a grid
a_files =[f for f in npy_files if f.endswith('_Q.npy')]

for file in a_files:
    first_image = np.load(file)
    channel_4 = first_image[3]

    # Use the actual min/max for better contrast
    img = plot_profiles(channel_4)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))
    plt.imshow(img_rgb, aspect='auto')
    plt.title(f"{os.path.basename(file)}")
    # plt.axis('off')
    plt.show()

    
letter = 'A'

#visualize all the A files in a grid
letter_files =[f for f in npy_files if f.endswith(f'_{letter}.npy')]

# Compute global min/max for all images in the grid
all_vals = []
for file in a_files:
    data = np.load(file)
    channel_4 = data[3]
    all_vals.append(channel_4)
all_vals = np.concatenate([np.load(f)[3].flatten() for f in letter_files])
global_min = np.percentile(all_vals, 0)
global_max = np.percentile(all_vals, 99.97)

# Now plot using the same min/max for all
for idx, file in enumerate(a_files):
    data = np.load(file)
    channel_4 = data[3]
    img = plot_profiles(channel_4, max_val=global_max, min_val=global_min)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # ... (rest of your plotting code)

    plt.figure(figsize=(8, 4))
    plt.imshow(img_rgb, aspect='auto')
    plt.title(f"{os.path.basename(file)}")
    # plt.axis('off')
    plt.show()

letter = 'A'
a_files = [f for f in npy_files if f.endswith(f'_{letter}.npy')]
n = len(a_files)
cols = 4
rows = math.ceil(n / cols)

# Compute global min/max for all images in the grid
all_vals = np.concatenate([np.load(f)[3].flatten() for f in a_files])
global_min = np.percentile(all_vals, 0)
global_max = np.percentile(all_vals, 99.97)  # Adjust as needed

# Get shape for aspect ratio
sample_data = np.load(a_files[0])
h, w = sample_data[3].shape
subplot_width = 4
subplot_height = subplot_width * h / w
fig_width = subplot_width * cols
fig_height = subplot_height * rows

fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

for idx, file in enumerate(a_files):
    data = np.load(file)
    channel_4 = data[3]
    img = plot_profiles(channel_4, max_val=global_max, min_val=global_min)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    row = idx // cols
    col = idx % cols
    ax = axes[row, col] if rows > 1 else axes[col]
    ax.imshow(img_rgb, aspect='auto')
    ax.set_title(os.path.basename(file))
    # ax.axis('off')

# Hide any unused subplots
for idx in range(n, rows*cols):
    row = idx // cols
    col = idx % cols
    ax = axes[row, col] if rows > 1 else axes[col]
    ax.axis('off')

plt.show()

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#visualize all the A files in a grid
for letter in letters:
    letter_files =[f for f in npy_files if f.endswith(f'_{letter}.npy')]
    
    n = len(letter_files)
    cols = 4
    rows = math.ceil(n / cols)

    # Compute global min/max for all images in the grid
    all_vals = np.concatenate([np.load(f)[3].flatten() for f in letter_files])
    global_min = np.percentile(all_vals, 0)
    global_max = np.percentile(all_vals, 99.97)  # Adjust as needed

    # Get shape for aspect ratio
    sample_data = np.load(letter_files[0])
    h, w = sample_data[3].shape
    subplot_width = 4
    subplot_height = subplot_width * h / w
    fig_width = subplot_width * cols
    fig_height = subplot_height * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    for idx, file in enumerate(letter_files):
        data = np.load(file)
        channel_4 = data[3]
        img = plot_profiles(channel_4, max_val=global_max, min_val=global_min)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.imshow(img_rgb, aspect='auto')
        ax.set_title(os.path.basename(file))
        # ax.axis('off')

    # Hide any unused subplots
    for idx in range(n, rows*cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.show()