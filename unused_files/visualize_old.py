import numpy as np

img_array = np.load('glasses_train_8/diff/acoustic_diff_0_Q.npy')

print("Shape of img_array:", img_array.shape)  

from matplotlib import pyplot as plt

import numpy as np
import glob
import os
from matplotlib import pyplot as plt

# Get all .npy files in the directory
npy_files = glob.glob('glasses_test_repeat2/diff/*.npy')

import numpy as np
import glob
import os
from matplotlib import pyplot as plt

# Get all .npy files in the directory
'''
npy_files = glob.glob('glasses_test_repeat2/diff/*.npy')

num_files = len(npy_files)
fig, axes = plt.subplots(num_files, 4, figsize=(16, 4 * num_files))

for row, file_path in enumerate(npy_files):
    img_array = np.load(file_path)
    for col in range(4):
        ax = axes[row, col] if num_files > 1 else axes[col]
        ax.imshow(img_array[col], cmap='gray')
        if col == 0:
            ax.set_ylabel(os.path.basename(file_path), rotation=0, labelpad=40, fontsize=10, va='center')
        ax.set_title(f"Channel {col}")
        ax.axis('off')

plt.tight_layout()
plt.show()'
'''

#have it show up one at a time, and the next one shows up when you close the window

for file_path in npy_files:
    img_array = np.load(file_path)
    print(f"File: {os.path.basename(file_path)}, Shape: {img_array.shape}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(img_array[i], cmap='gray')
        axes[i].set_title(f"Channel {i}")
        axes[i].axis('off')
    plt.suptitle(os.path.basename(file_path))
    plt.tight_layout()
    plt.show()


#visualize just one
# fig, axes = plt.subplots(1, 4, figsize=(16, 4))
# for i in range(4):
#     axes[i].imshow(img_array[i], cmap='gray')
#     axes[i].set_title(f"Channel {i}")
#     axes[i].axis('off')
# plt.tight_layout()
# plt.show()