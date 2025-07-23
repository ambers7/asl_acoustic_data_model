import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import cv2
import logging
import random
import re  # Add re import for regex
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from copy import deepcopy
from torch import Tensor
from math import ceil
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define our categories (only grammar and mouth morphemes)
lst = ["none", 
       "raise", "furrow", "shake",  # Grammar (3)
       "puff", "oo", "mm", "cha", "th", "cs"]  # Mouth morphemes (5)

# Create label mappings
label_dic = {value: index for index, value in enumerate(lst)}
label_dic_reverse = {index: value for index, value in enumerate(lst)}
class_num = len(lst)
num_folds = 5

parser = argparse.ArgumentParser(description='Conditions')
parser.add_argument('--dataset_path', default='', type=str, help='dataset')
parser.add_argument('-poi','--point_of_interest', default='0,600', type=str, help='point of interest')
parser.add_argument('-g','--gpu_num', default=0, type=int, help='gpus')
parser.add_argument('--target_height', default=80, type=int, help='target')
parser.add_argument('--epoch', default=100, type=int, help='epoch')
parser.add_argument('-rt','--retraining', default=False, type=bool, help='retraining')
parser.add_argument('--exclude_sessions', default='', type=str, help='exclude-session')
parser.add_argument('--test_sessions', default='', type=str, help='test-session')
parser.add_argument('--folder_name', default='', type=str, help='folder_name')
parser.add_argument('--batch', default=5, type=int, help='batch')
parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume from')

args = parser.parse_args()

def ensure_folder_exists(folder_path):
    """Check if a folder exists, and create it if not."""
    if not os.path.exists(folder_path):  # Check if folder exists
        os.makedirs(folder_path)  # Create folder (including parent directories if needed)
        print(f"✅ Folder created: {folder_path}")
    else:
        print(f"📂 Folder already exists: {folder_path}")

# Example usage
retraining = args.retraining
num_epochs = args.epoch
target_height = args.target_height
gpu_set = args.gpu_num
dataset_folder = args.dataset_path
poi = args.point_of_interest
poi_list = poi.split(',')
exclude_sessions = args.exclude_sessions
test_sessions = args.test_sessions
folder_nm = args.folder_name
batch_size = args.batch

fusion = False
imu_1d = False
only_imu = False

input_channel_slice = [0,1,2,3] #use channel 4 for acoustic data
input_channel = len(input_channel_slice)

folder = dataset_folder.split('/dataset/')[0]+'_poi_%s_%s'%(poi_list[0],poi_list[1])+'_th_%s'%(target_height)+'ch%s'%input_channel + '_fusion_%s'%folder_nm

best_save_path = "./experiments/%s/"%(folder)
ensure_folder_exists(best_save_path)

# Set up logging configuration
def setup_logging(log_file_path):
    """Set up logging configuration once."""
    # Clear any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),  # Print to console
            logging.FileHandler(log_file_path, mode='a')  # Log to file (append mode)
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging(best_save_path + "logfile.txt")

def print_and_log(message):
    """Print message to console and log it."""
    print(message)  # Print to console
    logger.info(message)  # Log the message

# Log script start
print_and_log("="*50)
print_and_log("Training script started")
print_and_log(f"Experiment folder: {best_save_path}")
print_and_log("="*50)

# Log the configuration
print_and_log("="*50)
print_and_log("Model Configuration:")
print_and_log(f"Number of classes: {class_num}")
print_and_log("\nClass categories:")
print_and_log("Grammar (1-3): " + ", ".join(lst[1:4]))
print_and_log("Mouth morphemes (4-10): " + ", ".join(lst[4:]))
print_and_log("="*50)

def save_checkpoint(model, optimizer, epoch, best_acc=0.0, filename=best_save_path + "best_checkpoint.pth"):
    """Save model, optimizer, epoch number, and best accuracy."""
    checkpoint = {
        "epoch": epoch,  
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_accuracy": best_acc
    }
    torch.save(checkpoint, filename)
    print_and_log(f"✅ Checkpoint saved at epoch {epoch+1} with best accuracy: {best_acc:.2f}%")

def collate_various_size(batch):
    data_list_arr = [x[0][0] for x in batch]
    # data_list_imu = [x[0][1] for x in batch]  # Not using IMU data
    target = [x[1] for x in batch]
    filenames = [x[2] for x in batch]  # Keep filenames
    data_max_size = max([x.shape[1] for x in data_list_arr])
    
    window_size = 10
    target_length = data_max_size 
    target_length = ceil(target_length / window_size) * window_size
   
    data_arr = np.zeros((len(batch), data_list_arr[0].shape[0], target_length, data_list_arr[0].shape[2]))
    
    # horizontal shifting time axis. 
    for i in range(0, len(data_list_arr)):
        start_x = random.randint(0, target_length - data_list_arr[i].shape[1])
        data_arr[i, :, start_x: start_x + data_list_arr[i].shape[1], :] = data_list_arr[i]

    data_arr = data_arr.swapaxes(2,3) # C, H (spatial height), W (temporal dimension, e.g., time steps)
        
    return (data_arr, None), target, filenames  # Return None for IMU data

class DataSplitter:
    train_loader: DataLoader
    val_loader: DataLoader

    def __init__(self, train_data, test_data, BATCH_SIZE, WORKER_NUM):
        train_data = train_data
        val_data = test_data

        print_and_log('train length: ' + str(len(train_data)))
        print_and_log('test length: ' + str(len(val_data)))
        
        if len(train_data):
            train_dataset = CNNDataset(train_data, is_train=True)
            self.train_loader = DataLoader(
                train_dataset,
                num_workers=WORKER_NUM,
                collate_fn=collate_various_size,
                batch_sampler=DataBatches(len(train_dataset), BATCH_SIZE)
            )
        else:
            self.train_loader = None
            
        test_dataset = CNNDataset(val_data, is_train=False)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKER_NUM,
            collate_fn=collate_various_size,
        )

class CNNDataset(torch.utils.data.Dataset):
    def __init__(self, data, is_train):
        self.data = data
        self.is_train = is_train
        
    def __getitem__(self, index):
        input_arr = self.data[index][0]
        # input_imu = self.data[index][2]  # Not using IMU data
        output_arr = deepcopy(self.data[index][1])
        filename = self.data[index][3]  # Keep filename

        input_arr_copy = deepcopy(input_arr)

        aug_arr = input_arr_copy

        if self.is_train:
            if (random.random() > 0.2):
                mask_width = random.randint(10, 20)
                rand_start = random.randint(0, aug_arr.shape[1] - mask_width)
                aug_arr[:, rand_start: rand_start + mask_width, :] = 0.0

        padded_input = aug_arr

        if self.is_train:
            if random.random() > 0.2:
                noise_arr = np.random.random(padded_input.shape).astype(np.float32) * 0.1 + 0.95
                padded_input *= noise_arr

        padded_input_list = []
        
        for j in range(0, padded_input.shape[0]):
            padded_input_tmp = padded_input[j]
            for c in range(padded_input_tmp.shape[0]):
                # instance-level norm
                mu, sigma = np.mean(padded_input_tmp[c]), np.std(padded_input_tmp[c])
                if sigma < 1e-8:
                    padded_input_tmp[c] = padded_input_tmp[c] - mu
                else:
                    padded_input_tmp[c] = (padded_input_tmp[c] - mu) / sigma

            padded_input_tmp = np.nan_to_num(padded_input_tmp, nan=0.0, posinf=0.0, neginf=0.0)
            padded_input_list.append(padded_input_tmp)

        padded_input_fn = np.array(padded_input_list)

        # if poi
        padded_input_fn = padded_input_fn[:,:,int(poi_list[0]):int(poi_list[1])]
        poi_length = int(poi_list[1]) - int(poi_list[0])
        
        if self.is_train:
            target_height_start = random.randint(0, poi_length-target_height)
            target_height_end = target_height_start + target_height
            padded_input_fn = padded_input_fn[:,:,target_height_start:target_height_end]
        else:
            # test_dataset
            padded_input_fn = padded_input_fn[:,:,:target_height]

        return (padded_input_fn, None), output_arr, filename  # Return None for IMU data

    def __len__(self):
        return len(self.data)
    
class DataBatches:
    def __init__(self, dataset_size, batch_size):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.all_indices = self.batches()
        random.shuffle(self.all_indices)
        
    def __len__(self):
        return ceil(self.dataset_size / self.batch_size)

    def __iter__(self):
        for x in self.all_indices:
            yield x
        random.shuffle(self.all_indices)

    def batches(self):
        all_indices = []
        for i in range(0, self.dataset_size, self.batch_size):
            all_indices += [list(range(i, min(i + self.batch_size, self.dataset_size)))]
        return all_indices

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

    # Create upsampled time array
    upsampled_time = np.linspace(unique_time[0], unique_time[-1], target_num_samples)

    # Interpolate IMU data at upsampled time points
    upsampled_imu_data = np.column_stack([f(upsampled_time) for f in interp_functions])

    return upsampled_time, upsampled_imu_data

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

def read_from_folder(session_num, data_path, is_train=False):
    # Construct file path correctly using os.path.join
    file_path = os.path.join(data_path, f"session_{session_num}")
    file_echo_org = os.path.join(file_path, 'acoustic', 'non_diff')
    file_echo_diff = os.path.join(file_path, 'acoustic', 'diff')
    file_imus = os.path.join(file_path, 'imu')
    file_echo_org_list = sorted([f for f in os.listdir(file_echo_org)])
    file_echo_diff_list = sorted([f for f in os.listdir(file_echo_diff)])
    file_imus_list = sorted([f for f in os.listdir(file_imus)])

    data_pairs = []
    n_bad = 0
    bad_signal_remove_length = 5
    
    # Initialize counters for each category
    category_counts = {label: 0 for label in lst}
    
    print_and_log(f"\nProcessing files in session {session_num}:")
    
    for i in range(0, len(file_echo_diff_list)):
        file = file_echo_diff_list[i]
        # print_and_log(f"Processing file: {file}")  # Debug print
        
        # Extract label from filename (e.g., 'acoustic_diff_9_dorm(cs).npy' or 'acoustic_diff_9_dorm(none).npy')
        match = re.search(r'\((.*?)\)', file)
        if not match:
            print_and_log(f"Warning: No label found in parentheses for file {file}")
            continue
            
        truth = match.group(1).lower()  # Get text between parentheses and convert to lowercase
        # print_and_log(f"Found label: {truth}")  # Debug print

        # Skip if the label is not in our defined categories
        if truth not in lst:
            print_and_log(f"Warning: Skipping unknown label '{truth}' from file {file}")
            continue

        # Load imu
        try:
            File_data = np.loadtxt(file_imus+"/"+file_imus_list[i], dtype=str, delimiter=" ") 
            all_imu = np.array(File_data, dtype=float)[:, :3]
            all_imu_time = np.array(File_data, dtype=float)[:, 3:]
            all_imu_time = np.array([i[0] for i in all_imu_time])
           
            # Load echo_diff
            profiles = np.load(file_echo_diff+"/"+file_echo_diff_list[i])
            profile_data_piece = profiles.copy()
            profile_data_piece = profile_data_piece.swapaxes(1, 2)

            # Upsampling imu data based on echo profile
            psampled_time, upsampled_imu_data = upsample_imu_data(all_imu_time, all_imu, profile_data_piece.shape[1])
            normalized_imu_data, means, stds = normalize_imu_data(upsampled_imu_data)
            normalized_imu_data.shape = 1, normalized_imu_data.shape[0], normalized_imu_data.shape[1]

            if profile_data_piece.shape[1] > 50:  # check the data quality 
                data_pairs += [(profile_data_piece[:,:-bad_signal_remove_length,:], 
                              truth, 
                            #   normalized_imu_data[:,:-bad_signal_remove_length,:])
                              all_imu.reshape(1, all_imu.shape[0], all_imu.shape[1]),
                              file)] #add filename
                category_counts[truth] += 1
                # print_and_log(f"Successfully added {truth} sample")  # Debug print
            else:
                n_bad += 1
                print_and_log(f"Skipped due to quality check (length <= 50)")
        except Exception as e:
            print_and_log(f"Error processing file {file}: {str(e)}")
            continue

    # Print category statistics
    print_and_log("\nCategory distribution for session %s:" % session_num)
    print_and_log("-" * 40)
    
    # Print grammar signs
    print_and_log("Grammar signs:")
    for label in lst[1:4]:  # raise, furrow, shake
        print_and_log(f"  {label}: {category_counts[label]}")
    
    # Print mouth morphemes
    print_and_log("\nMouth morphemes:")
    for label in lst[4:]:  # puff, oo, mm, cha, th, cs
        print_and_log(f"  {label}: {category_counts[label]}")
    
    # Print none category
    print_and_log("\nNone category:")
    print_and_log(f"  none: {category_counts['none']}")
    
    print_and_log("-" * 40)
    if n_bad:
        print_and_log('     %d bad data pieces' % n_bad)

    return data_pairs, []  # Return empty list for loaded_gt since we don't use it

def save_cm_figure(true_label, predict_label, best_save_path, acc, lst): 
    # Convert numeric labels to class names
    true_labels = [label_dic_reverse[i] for i in true_label]
    predicted_labels = [label_dic_reverse[i] for i in predict_label]
    
    # Get unique class names and sort them to maintain order
    unique_classes = lst  # Use our predefined list to maintain category grouping
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Create figure
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    
    # Customize labels
    plt.xticks(ticks=np.arange(len(unique_classes)) + 0.5, labels=unique_classes, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(unique_classes)) + 0.5, labels=unique_classes, rotation=0)
    
    # Add titles and labels
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Best Accuracy: {acc:.2f}%")
    
    # Save with high quality
    plt.savefig(best_save_path + "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

def get_category_stats(data):
    """Get statistics about category distribution in the data."""
    stats = defaultdict(int)
    for item in data:
        label = item[1]
        if label == "none":
            stats['None'] += 1
            stats[f"None - {label}"] += 1
        elif label in lst[1:4]:  # Grammar signs
            stats['Grammar'] += 1
            stats[f"Grammar - {label}"] += 1
        elif label in lst[4:]:  # Mouth morphemes
            stats['Mouth'] += 1
            stats[f"Mouth - {label}"] += 1
    return stats

def create_category_folds(category_data, n_folds):
    """Split category data into n_folds while maintaining label distribution within the category."""
    # Create label arrays for stratification
    labels = [label_dic[x[1]] for x in category_data]
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Create folds
    folds = []
    indices = list(range(len(category_data)))
    for _, test_idx in skf.split(indices, labels):
        folds.append([category_data[i] for i in test_idx])
    
    return folds

def create_stratified_folds(data_path, n_folds=num_folds):
    """Create stratified k-folds ensuring balanced distribution of grammar and mouth morpheme signs."""
    # First, organize data by category
    none_data = []
    grammar_data = []
    mouth_data = []
    
    # Read all files and categorize them
    dataset_path = os.path.join(data_path, 'dataset')
    for session in os.listdir(dataset_path):
        if session.startswith('session_'):
            session_num = session.split('_')[1]  # Get just the number part
            data_pairs, _ = read_from_folder(session_num, dataset_path, is_train=True)
            
            # Categorize each sample
            for data in data_pairs:
                label = data[1]  # Get the label
                if label == "none":
                    none_data.append(data)
                elif label in lst[1:4]:  # Grammar signs (raise, furrow, shake)
                    grammar_data.append(data)
                elif label in lst[4:]:  # Mouth morphemes (puff, oo, mm, cha, th, cs)
                    mouth_data.append(data)
    
    print_and_log(f"Total samples - None: {len(none_data)}, Grammar: {len(grammar_data)}, Mouth: {len(mouth_data)}")
    
    # Create stratified folds for each category
    none_folds = create_category_folds(none_data, n_folds)
    grammar_folds = create_category_folds(grammar_data, n_folds)
    mouth_folds = create_category_folds(mouth_data, n_folds)
    
    # Combine folds while maintaining stratification
    combined_folds = []
    for i in range(n_folds):
        train_data = []
        test_data = []
        
        # Add none samples
        for j in range(n_folds):
            if j != i:  # Training fold
                train_data.extend(none_folds[j])
            else:  # Test fold
                test_data.extend(none_folds[j])
        
        # Add grammar samples
        for j in range(n_folds):
            if j != i:  # Training fold
                train_data.extend(grammar_folds[j])
            else:  # Test fold
                test_data.extend(grammar_folds[j])
        
        # Add mouth morpheme samples
        for j in range(n_folds):
            if j != i:  # Training fold
                train_data.extend(mouth_folds[j])
            else:  # Test fold
                test_data.extend(mouth_folds[j])
        
        # Shuffle the combined data
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        combined_folds.append((train_data, test_data))
        
        # Log fold statistics
        train_stats = get_category_stats(train_data)
        test_stats = get_category_stats(test_data)
        print_and_log(f"\nFold {i+1} statistics:")
        print_and_log("Training set:")
        if 'None' in train_stats:
            print_and_log(f"  None: {train_stats['None']}")
        if 'Grammar' in train_stats:
            print_and_log(f"  Grammar: {train_stats['Grammar']}")
        if 'Mouth' in train_stats:
            print_and_log(f"  Mouth: {train_stats['Mouth']}")
        for cat, count in train_stats.items():
            if ' - ' in cat:  # Individual category counts
                print_and_log(f"  {cat}: {count}")
        print_and_log("Test set:")
        if 'None' in test_stats:
            print_and_log(f"  None: {test_stats['None']}")
        if 'Grammar' in test_stats:
            print_and_log(f"  Grammar: {test_stats['Grammar']}")
        if 'Mouth' in test_stats:
            print_and_log(f"  Mouth: {test_stats['Mouth']}")
        for cat, count in test_stats.items():
            if ' - ' in cat:  # Individual category counts
                print_and_log(f"  {cat}: {count}")
    
    return combined_folds

# Create stratified folds
folds = create_stratified_folds(args.dataset_path)

# Dictionary to store results for each fold
fold_results = {}

# Lists to store all predictions and true labels across folds
all_predictions = []
all_true_labels = []

# Handle resume functionality - determine starting fold
start_fold = 0
if args.resume:
    if not os.path.isfile(args.resume):
        print_and_log(f"❌ Resume checkpoint not found: {args.resume}")
        args.resume = None

    # Extract fold number from checkpoint path
    fold_match = re.search(r'fold[_\-]?(\d+)', args.resume)

    if fold_match:
        start_fold = int(fold_match.group(1)) - 1  # Convert to 0-based index
        print_and_log(f"Will resume training from fold {start_fold + 1}")
    else:
        print_and_log("⚠️ Could not determine fold number from checkpoint path, starting from fold 1")


# Train on each fold
for current_fold in range(num_folds):
    # Skip folds we've already completed when resuming
    if current_fold < start_fold:
        print_and_log(f"Skipping fold {current_fold + 1} (already completed)")
        continue

    print_and_log("="*50)
    print_and_log(f"Training on fold {current_fold + 1}/{num_folds}")
    print_and_log("Note: Data is stratified by grammar and mouth morpheme categories")
    
    # Get data for current fold
    train_data, test_data = folds[current_fold]
    
    # Create data loaders
    data_splitter = DataSplitter(train_data, test_data, batch_size, 0)
    train_loader = data_splitter.train_loader
    test_loader = data_splitter.test_loader
    
    # Initialize model for this fold
    model = models.resnet18(num_classes=class_num)
    model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
    device = torch.device("cuda:%d"%gpu_set if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize tracking variables
    best_val_acc = 0.0
    fold_dir = os.path.join(best_save_path, f"fold_{current_fold+1}")
    fold_checkpoint_path = os.path.join(best_save_path, f"fold_{current_fold+1}_best_checkpoint.pth")
    best_predictions = None
    best_true_labels = None
    best_filenames = None # Initialize best_filenames
    
    # Handle resume functionality for this fold
    start_epoch = 0
    if args.resume and current_fold == start_fold:
        print_and_log(f"Resuming fold {current_fold + 1} from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_accuracy', 0.0)
        print_and_log(f"Resumed from epoch {start_epoch}")
        print_and_log(f"Training will continue from epoch {start_epoch} to {num_epochs}")
        print_and_log(f"📊 Loaded previous best accuracy: {best_val_acc:.2f}%")
    else:
        print_and_log(f"Starting training from epoch 0 to {num_epochs}")
   
   # Training loop for this fold
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training step
        for i, (input_arr_raw, target, filename) in enumerate(train_loader):
            input_arr = input_arr_raw[0][:,input_channel_slice,:,:]
            # input_imu = input_arr_raw[1][:,:,:,:] # Not using IMU data
            
            if not isinstance(input_arr, torch.Tensor):
                input_arr = Tensor(input_arr).to(device)
            else:
                input_arr = input_arr.to(device)
                
            labels = torch.tensor([label_dic[x] for x in target], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(input_arr)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Validation step
        if epoch % 3 == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            predictions = []
            true_labels = []
            filenames = []  # Store filenames
            
            with torch.no_grad():
                for i, (input_arr_raw, target, filename) in enumerate(test_loader):  # Get filename
                    input_arr = input_arr_raw[0][:,input_channel_slice,:,:]
                    # input_imu = input_arr_raw[1][:,:,:,:] # Not using IMU data
                    
                    input_arr = Tensor(input_arr).to(device)
                    labels = torch.tensor([label_dic[x] for x in target], dtype=torch.long).to(device)
                    
                    outputs = model(input_arr)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    
                    predictions.extend(predicted.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                    filenames.extend(filename)  # Store filenames
            
            test_acc = 100 * test_correct / test_total
            
            if test_acc > best_val_acc:
                best_val_acc = test_acc
                # Save fold-specific checkpoint
                save_checkpoint(model, optimizer, epoch, best_acc=best_val_acc, 
                             filename=fold_checkpoint_path)
                # Save fold-specific confusion matrix
                save_cm_figure(true_labels, predictions, 
                             os.path.join(fold_dir, f"confusion_matrix.png"),
                             best_val_acc, lst)
                
                # Save fold-specific test results with all test cases
                test_results = []
                for filename, true_label, pred_label in zip(filenames, true_labels, predictions):
                    # Extract sign name from filename (e.g., "black(oo)" from "acoustic_diff_black(oo).npy")
                    sign_name = filename.split('_')[-1].split('.')[0]  # Get last part before .npy
                    test_results.append({
                        'Sign': sign_name,
                        'Truth': label_dic_reverse[int(true_label)],
                        'Predicted': label_dic_reverse[int(pred_label)],
                        'Fold': f'Fold_{current_fold + 1}'
                    })
                results_df = pd.DataFrame(test_results)
                results_df.to_csv(os.path.join(fold_dir, f"results.csv"), index=False)

            
            print_and_log(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
                         f"Train Accuracy: {100 * correct/total:.2f}%, "
                         f"Test Accuracy: {test_acc:.2f}%, Best: {best_val_acc:.2f}%")
    
    
    
    # Clear GPU memory
    del model
    del optimizer
    torch.cuda.empty_cache()

# Print summary of all folds
print_and_log("\n" + "="*50)
print_and_log("Training Complete - Summary of All Folds")
print_and_log("="*50)
total_acc = 0
for fold_num, results in fold_results.items():
    print_and_log(f"Fold {fold_num + 1}: Best Accuracy = {results['best_accuracy']:.2f}%")
    total_acc += results['best_accuracy']
print_and_log("-"*50)
print_and_log(f"Average Accuracy Across All Folds: {total_acc/num_folds:.2f}%")
print_and_log("="*50)

# Create combined confusion matrix from all folds' best results
all_predictions = []
all_true_labels = []
for fold_num, results in fold_results.items():
    all_predictions.extend(results['predictions'])
    all_true_labels.extend(results['true_labels'])

# Save combined confusion matrix
save_cm_figure(all_true_labels, all_predictions,
               os.path.join(best_save_path, "confusion_matrix_combined.png"),
               total_acc/num_folds, lst)

print_and_log("\nCreated combined confusion matrix from all folds")
print_and_log("="*50)

test_results = []
for fold_num, results in fold_results.items():
    if 'true_labels' in results and 'predictions' in results:
        true_labels = [label_dic_reverse[int(label)] for label in results['true_labels']]
        predicted_labels = [label_dic_reverse[int(pred)] for pred in results['predictions']]

        # true_labels = [label_dic_reverse[label] for label in results['true_labels']]
        # predicted_labels = [label_dic_reverse[pred] for pred in results['predictions']]
        
        for true_label, pred_label in zip(true_labels, predicted_labels):
            test_results.append({
                'True Label': true_label,
                'Predicted Label': pred_label,
                'Fold': f'Fold_{fold_num + 1}'
            })
    else:
        print_and_log(f"⚠️  Fold {fold_num + 1} missing predictions, skipped in test_results.csv")

# Save to CSV
results_df = pd.DataFrame(test_results)
results_df.to_csv(os.path.join(best_save_path, "test_results.csv"), index=False)
print_and_log("\nSaved detailed test results to test_results.csv")
print_and_log("="*50)