'''
python test_model.py --model_path ./experiments/data/asl_test_poi_250_600_th_330ch1_fusion_gnd_trth_all_signs/best_model.pth 
--test_datasets "10_01,11_01"

python test_model.py --model_path ./experiments/poi_0_600_th_80ch1_fusion_/best_model.pth

python test_model.py \
    --model_path ./experiments/poi_0_600_th_80ch1_fusion_/best_model.pth \
    --test_datasets "10_01,11_01" \
    --batch_size 10 \
    --gpu_num 0 \
    --output_dir ./my_test_results/


python test_model.py \
    --model_path ./experiments/poi_0_600_th_80ch1_fusion_/best_model.pth \
    --input_channel 3 \
    --test_datasets "10_01"
'''

import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
import logging
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import random
from math import ceil

# Import the same model classes from emo_test_cnn.py
import sys
sys.path.append('.')

# Define the needed functions and classes directly to avoid import conflicts
def upsample_imu_data(time, imu_data, target_num_samples):
    """Upsample IMU data to a target number of samples."""
    import numpy as np
    
    # Ensure time values are strictly increasing and remove duplicates
    unique_time, unique_idx = np.unique(time, return_index=True)
    sorted_idx = np.argsort(unique_time)
    unique_time = unique_time[sorted_idx]
    unique_idx = unique_idx[sorted_idx]

    # Sort imu_data based on unique_time
    sorted_imu_data = imu_data[unique_idx]

    # Create an interpolation function for each dimension of the IMU data
    interp_functions = [CubicSpline(unique_time, sorted_imu_data[:, i]) for i in range(sorted_imu_data.shape[1])]

    # Create upsampled time array
    upsampled_time = np.linspace(unique_time[0], unique_time[-1], target_num_samples)

    # Interpolate IMU data at upsampled time points
    upsampled_imu_data = np.column_stack([f(upsampled_time) for f in interp_functions])

    return upsampled_time, upsampled_imu_data

def normalize_imu_data(upsampled_imu_data):
    """Normalize upsampled IMU data."""
    import numpy as np
    
    means = np.mean(upsampled_imu_data, axis=0)
    stds = np.std(upsampled_imu_data, axis=0)

    normalized_imu_data = (upsampled_imu_data - means) / stds

    return normalized_imu_data, means, stds

def read_from_folder(session_num, data_path, is_train=False):
    """Read data from folder - simplified version for testing."""
    import os
    import numpy as np
    
    file_path = data_path + '%s'%str(session_num)
    file_echo_diff = file_path +  "/" + 'acoustic/diff'
    file_imus = file_path +  "/"  + 'imu'
    
    file_echo_diff_list = sorted([f for f in os.listdir(file_echo_diff)])
    file_imus_list = sorted([f for f in os.listdir(file_imus)])

    loaded_gt = []
    data_pairs = []
    n_bad = 0
    bad_signal_remove_length = 5
    
    for i in range(0, len(file_echo_diff_list)):
        # Extract label from filename
        file = file_echo_diff_list[i]
        truth = file.split('_')[-1].split('.')[0]

        # Load imu
        File_data = np.loadtxt(file_imus+"/"+file_imus_list[i], dtype=str, delimiter=" ") 
        all_imu = np.array(File_data, dtype=float)[:, :3]
        all_imu_time = np.array(File_data, dtype=float)[:, 3:]
        all_imu_time = np.array([i[0] for i in all_imu_time])
       
        # Load echo_diff
        profiles = np.load(file_echo_diff+"/"+file_echo_diff_list[i])
        profile_data_piece = profiles.copy()
        profile_data_piece = profile_data_piece.swapaxes(1, 2) # 

        # upsampling imu data based on echo profile
        upsampled_time, upsampled_imu_data = upsample_imu_data(all_imu_time, all_imu, profile_data_piece.shape[1])
        normalized_imu_data, means, stds = normalize_imu_data(upsampled_imu_data)
        normalized_imu_data.shape = 1, normalized_imu_data.shape[0], normalized_imu_data.shape[1]

        if profile_data_piece.shape[1] > 50: # check the data quality 
            if truth in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']:
                data_pairs += [(profile_data_piece[:,:-bad_signal_remove_length,:], 
                                truth, 
                                normalized_imu_data[:,:-bad_signal_remove_length,:])]
        else:
            n_bad +=1

    if n_bad:
        print('     %d bad data pieces' % n_bad)

    return data_pairs, loaded_gt

# Define the model classes directly to avoid import conflicts
class BasicBlock(torch.nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = torch.nn.functional.relu(out)
        return out

class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

class CNNDataset(torch.utils.data.Dataset):
    def __init__(self, data, is_train):
        self.data = data
        self.is_train = is_train
        
    def __getitem__(self, index):
        input_arr = self.data[index][0]
        input_imu = self.data[index][2]
        output_arr = deepcopy(self.data[index][1])

        input_arr_copy = deepcopy(input_arr)
        input_imu_copy = deepcopy(input_imu)

        aug_arr = input_arr_copy
        aug_imu = input_imu_copy

        if self.is_train:
            if (random.random() > 0.2):
                mask_width = random.randint(10, 20)
                rand_start = random.randint(0, aug_arr.shape[1] - mask_width)
                aug_arr[:, rand_start: rand_start + mask_width, :] = 0.0
                aug_imu[:, rand_start: rand_start + mask_width, :] = 0.0

        padded_input = aug_arr
        padded_imu = aug_imu

        if self.is_train:
            if random.random() > 0.2:
                noise_arr = np.random.random(padded_input.shape).astype(np.float32) * 0.1 + 0.95
                noise_imu = np.random.random(padded_imu.shape).astype(np.float32) * 0.1 + 0.95
                padded_input *= noise_arr
                padded_imu *= noise_imu

        padded_input_list = []
        
        for j in range(0, padded_input.shape[0]):
            padded_input_tmp = padded_input[j]
            for c in range(padded_input_tmp.shape[0]):
                mu, sigma = np.mean(padded_input_tmp[c]), np.std(padded_input_tmp[c])
                if sigma < 1e-8:
                    padded_input_tmp[c] = padded_input_tmp[c] - mu
                else:
                    padded_input_tmp[c] = (padded_input_tmp[c] - mu) / sigma

            padded_input_tmp = np.nan_to_num(padded_input_tmp, nan=0.0, posinf=0.0, neginf=0.0)
            padded_input_list.append(padded_input_tmp)

        padded_input_fn = np.array(padded_input_list)
        padded_imu = np.nan_to_num(padded_imu, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to tensors
        padded_input_fn = torch.tensor(padded_input_fn, dtype=torch.float32)
        padded_imu = torch.tensor(padded_imu, dtype=torch.float32)

        return (padded_input_fn, padded_imu), output_arr

    def __len__(self):
        return len(self.data)

def collate_various_size(batch):
    data_list_arr = [x[0][0] for x in batch]
    # data_list_imu = [x[0][1] for x in batch]  # Ignore IMU data
    target = [x[1] for x in batch]
    
    # Get max size for acoustic data only
    data_max_size = max([x.shape[1] for x in data_list_arr])
    
    window_size = 10
    target_length = data_max_size 
    target_length = ceil(target_length / window_size) * window_size

    data_arr = np.zeros((len(batch), data_list_arr[0].shape[0], target_length, data_list_arr[0].shape[2]))
    # data_imu = np.zeros((len(batch), data_list_imu[0].shape[0], target_length, data_list_imu[0].shape[2]))  # Ignore IMU
    
    for i in range(0, len(data_list_arr)):
        start_x = random.randint(0, target_length - data_list_arr[i].shape[1])
        data_arr[i, :, start_x: start_x + data_list_arr[i].shape[1], :] = data_list_arr[i]
        # data_imu[i, :, start_x: start_x + data_imu[i].shape[1], :] = data_list_imu[i]

    data_arr = data_arr.swapaxes(2,3)
    # data_imu = data_imu.swapaxes(2,3)  # Ignore IMU
        
    return (data_arr, None), target  # Return None for IMU data

parser = argparse.ArgumentParser(description='Test Model on Multiple Datasets')
parser.add_argument('--model_path', required=True, type=str, help='Path to the trained model (.pth file)')
parser.add_argument('--test_datasets', default='', type=str, help='Comma-separated list of test session folders (e.g., 0901,1001)')
parser.add_argument('--dataset_root', default='/data/asl_test', type=str, help='Root path to dataset folder')
parser.add_argument('--poi', default='0,600', type=str, help='Point of interest (start,end)')
parser.add_argument('--target_height', default=80, type=int, help='Target height for cropping')
parser.add_argument('--batch_size', default=5, type=int, help='Batch size for testing')
parser.add_argument('--gpu_num', default=0, type=int, help='GPU number to use')
parser.add_argument('--input_channel', default=3, type=int, help='Input channel slice')
parser.add_argument('--output_dir', default='./test_results/', type=str, help='Output directory for results')

args = parser.parse_args()

def ensure_folder_exists(folder_path):
    """Check if a folder exists, and create it if not."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✅ Folder created: {folder_path}")
    else:
        print(f"📂 Folder already exists: {folder_path}")

def print_and_log(message, log_file):
    """Print message to console and log it to a .txt file."""
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='a')
        ]
    )
    print(message)
    logging.info(message)

def save_cm_figure(true_label, predict_label, save_path, acc, lst):
    """Save confusion matrix figure."""
    true_labels = [label_dic_reverse[i] for i in true_label]
    predicted_labels = [label_dic_reverse[i] for i in predict_label]
    
    unique_classes = sorted(set(true_labels) | set(predicted_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    plt.xticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=90)
    plt.yticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=0)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Accuracy: {acc:.3f}%")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig(save_path + "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

# Setup
device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
ensure_folder_exists(args.output_dir)

# Class labels (same as emo_test_cnn.py)
lst = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
label_dic = {value: index for index, value in enumerate(lst)}
label_dic_reverse = {index: value for index, value in enumerate(lst)}
class_num = len(lst)

# Parse arguments
poi_list = args.poi.split(',')
input_channel_slice = [args.input_channel]
input_channel = len(input_channel_slice)

# Get test datasets
if args.test_datasets:
    test_sessions = [s.strip() for s in args.test_datasets.split(',')]
else:
    # If no specific sessions provided, test on all available
    dp = args.dataset_root + '/dataset/'
    tmp_dp = sorted(os.listdir(dp))
    test_sessions = [i.split('_')[1] for i in tmp_dp if i.find('session') == 0]

print(f"Testing on sessions: {test_sessions}")

# Load model
def load_model(model_path):
    """Load the appropriate model based on configuration."""
    model = resnet18(num_classes=class_num)
    model.conv1 = torch.nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load the model
print(f"Loading model from: {args.model_path}")
model = load_model(args.model_path)
print("✅ Model loaded successfully")

# Test on each dataset
results = {}
# Use the same path construction as emo_test_cnn.py
dp = args.dataset_root + '/dataset/'
data_path = dp + 'session_'

for session in test_sessions:
    print(f"\n{'='*50}")
    print(f"Testing on session: {session}")
    print(f"{'='*50}")
    
    # Load test data
    test_data, _ = read_from_folder(session, data_path, is_train=False)
    
    if len(test_data) == 0:
        print(f"⚠ No data found for session {session}")
        continue
    
    # Create dataset and dataloader
    test_dataset = CNNDataset(test_data, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_various_size
    )
    
    # Test the model
    model.eval()
    test_correct = 0
    test_total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for i, (input_arr_raw, target) in enumerate(test_loader):
            input_arr = input_arr_raw[0][:, input_channel_slice, :, :]
            # input_imu = input_arr_raw[1][:, :, :, :]  # Ignore IMU data
            
            # Convert to tensors
            if not isinstance(input_arr, torch.Tensor):
                input_arr = torch.tensor(input_arr, dtype=torch.float32).to(device)
            else:
                input_arr = input_arr.to(device)
            
            # if not isinstance(input_imu, torch.Tensor):
            #     input_imu = torch.tensor(input_imu, dtype=torch.float32).to(device)
            # else:
            #     input_imu = input_imu.to(device)
            
            labels = torch.tensor([label_dic[x] for x in target], dtype=torch.long).to(device)
            
            # Forward pass - only use acoustic data
            outputs = model(input_arr)
            
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    test_acc = 100 * test_correct / test_total
    results[session] = {
        'accuracy': test_acc,
        'correct': test_correct,
        'total': test_total,
        'predictions': predictions,
        'true_labels': true_labels
    }
    
    print(f"Session {session} - Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
    
    # Save individual session results
    session_dir = os.path.join(args.output_dir, f"session_{session}")
    ensure_folder_exists(session_dir)
    
    # Save confusion matrix
    save_cm_figure(true_labels, predictions, session_dir + "/", test_acc, lst)
    
    # Save detailed results
    df = pd.DataFrame({
        "True Label": [label_dic_reverse[i] for i in true_labels],
        "Predicted Label": [label_dic_reverse[i] for i in predictions],
        "True Index": true_labels,
        "Predicted Index": predictions
    })
    df.to_csv(session_dir + "/test_results.csv", index=False)
    
    # Save summary
    with open(session_dir + "/summary.txt", 'w') as f:
        f.write(f"Session: {session}\n")
        f.write(f"Accuracy: {test_acc:.2f}%\n")
        f.write(f"Correct: {test_correct}\n")
        f.write(f"Total: {test_total}\n")

# Overall summary
print(f"\n{'='*50}")
print("OVERALL RESULTS")
print(f"{'='*50}")

total_correct = sum(r['correct'] for r in results.values())
total_samples = sum(r['total'] for r in results.values())
overall_acc = 100 * total_correct / total_samples if total_samples > 0 else 0

print(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_samples})")

# Save overall results
overall_predictions = []
overall_true_labels = []
for session_data in results.values():
    overall_predictions.extend(session_data['predictions'])
    overall_true_labels.extend(session_data['true_labels'])

# Save overall confusion matrix
save_cm_figure(overall_true_labels, overall_predictions, args.output_dir, overall_acc, lst)

# Save overall results CSV
overall_df = pd.DataFrame({
    "True Label": [label_dic_reverse[i] for i in overall_true_labels],
    "Predicted Label": [label_dic_reverse[i] for i in overall_predictions],
    "True Index": overall_true_labels,
    "Predicted Index": overall_predictions
})
overall_df.to_csv(args.output_dir + "/overall_test_results.csv", index=False)

# Save summary report
with open(args.output_dir + "/summary_report.txt", 'w') as f:
    f.write("MODEL TESTING SUMMARY REPORT\n")
    f.write("=" * 50 + "\n")
    f.write(f"Model Path: {args.model_path}\n")
    f.write(f"Test Sessions: {', '.join(test_sessions)}\n")
    f.write(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_samples})\n\n")
    
    f.write("Individual Session Results:\n")
    f.write("-" * 30 + "\n")
    for session, data in results.items():
        f.write(f"Session {session}: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})\n")

print(f"\n✅ All results saved to: {args.output_dir}")
print(f"📊 Overall accuracy: {overall_acc:.2f}%")
