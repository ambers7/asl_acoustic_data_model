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

def ensure_folder_exists(folder_path):
    """Check if a folder exists, and create it if not."""
    if not os.path.exists(folder_path):  # Check if folder exists
        os.makedirs(folder_path)  # Create folder (including parent directories if needed)
        print(f"✅ Folder created: {folder_path}")
    else:
        print(f"📂 Folder already exists: {folder_path}")

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
        padded_input_fn = padded_input_fn[:,:,300:360]  # Hardcoded POI values from original training
        poi_length = 60  # 360 - 300
        
        if self.is_train:
            target_height_start = random.randint(0, poi_length-50)  # Hardcoded target height from original training
            target_height_end = target_height_start + 50
            padded_input_fn = padded_input_fn[:,:,target_height_start:target_height_end]
        else:
            # test_dataset
            padded_input_fn = padded_input_fn[:,:,:50]  # Hardcoded target height

        return (padded_input_fn, None), output_arr, filename  # Return None for IMU data

    def __len__(self):
        return len(self.data)

def upsample_imu_data(time, imu_data, target_num_samples):
    """
    Upsample IMU data to a target number of samples.
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
    """
    means = np.mean(upsampled_imu_data, axis=0)
    stds = np.std(upsampled_imu_data, axis=0)
    normalized_imu_data = (upsampled_imu_data - means) / stds
    return normalized_imu_data, means, stds

def read_from_folder(session_num, data_path, is_train=False):
    # Skip session 0301
    if session_num == '0301':
        print(f"Skipping excluded session {session_num}")
        return [], []

    file_path = data_path + '%s'%str(session_num)
    file_echo_org = file_path +  "/" + 'acoustic/non_diff'
    file_echo_diff = file_path +  "/" + 'acoustic/diff'
    file_imus = file_path +  "/"  + 'imu'
    file_echo_org_list = sorted([f for f in os.listdir(file_echo_org)])
    file_echo_diff_list = sorted([f for f in os.listdir(file_echo_diff)])
    file_imus_list = sorted([f for f in os.listdir(file_imus)])

    data_pairs = []
    n_bad = 0
    bad_signal_remove_length = 5
    
    # Initialize counters for each category
    category_counts = {label: 0 for label in lst}
    
    print(f"\nProcessing files in session {session_num}:")
    
    for i in range(0, len(file_echo_diff_list)):
        file = file_echo_diff_list[i]
        
        # Extract label from filename (e.g., 'acoustic_diff_9_dorm(cs).npy' or 'acoustic_diff_9_dorm(none).npy')
        match = re.search(r'\((.*?)\)', file)
        if not match:
            print(f"Warning: No label found in parentheses for file {file}")
            continue
            
        truth = match.group(1).lower()  # Get text between parentheses and convert to lowercase

        # Skip if the label is not in our defined categories
        if truth not in lst:
            print(f"Warning: Skipping unknown label '{truth}' from file {file}")
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
                              all_imu.reshape(1, all_imu.shape[0], all_imu.shape[1]),
                              file)] #add filename
                category_counts[truth] += 1
            else:
                n_bad += 1
                print(f"Skipped due to quality check (length <= 50)")
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    # Print category statistics
    print("\nCategory distribution for session %s:" % session_num)
    print("-" * 40)
    
    # Print grammar signs
    print("Grammar signs:")
    for label in lst[1:4]:  # raise, furrow, shake
        print(f"  {label}: {category_counts[label]}")
    
    # Print mouth morphemes
    print("\nMouth morphemes:")
    for label in lst[4:]:  # puff, oo, mm, cha, th, cs
        print(f"  {label}: {category_counts[label]}")
    
    # Print none category
    print("\nNone category:")
    print(f"  none: {category_counts['none']}")
    
    print("-" * 40)
    if n_bad:
        print('     %d bad data pieces' % n_bad)

    return data_pairs, []  # Return empty list for loaded_gt since we don't use it

def create_stratified_folds(data_path, n_folds=6):
    """Create stratified k-folds ensuring balanced distribution of grammar and mouth morpheme signs."""
    # First, organize data by category
    none_data = []
    grammar_data = []
    mouth_data = []
    
    # Read all files and categorize them
    session_path = data_path + '/dataset/session_'
    for session in os.listdir(data_path + '/dataset/'):
        if session.startswith('session_'):
            session_num = session.split('_')[1]
            data_pairs, _ = read_from_folder(session_num, data_path + '/dataset/session_', is_train=True)
            
            # Categorize each sample
            for data in data_pairs:
                label = data[1]  # Get the label
                if label == "none":
                    none_data.append(data)
                elif label in lst[1:4]:  # Grammar signs (raise, furrow, shake)
                    grammar_data.append(data)
                elif label in lst[4:]:  # Mouth morphemes (puff, oo, mm, cha, th, cs)
                    mouth_data.append(data)
    
    print(f"Total samples - None: {len(none_data)}, Grammar: {len(grammar_data)}, Mouth: {len(mouth_data)}")
    
    # Create stratified folds for each category
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

        train_stats = get_category_stats(train_data)
        test_stats = get_category_stats(test_data)
        print(f"\nFold {i+1} statistics:")
        print("Training set:")
        if 'None' in train_stats:
            print(f"  None: {train_stats['None']}")
        if 'Grammar' in train_stats:
            print(f"  Grammar: {train_stats['Grammar']}")
        if 'Mouth' in train_stats:
            print(f"  Mouth: {train_stats['Mouth']}")
        for cat, count in train_stats.items():
            if ' - ' in cat:  # Individual category counts
                print(f"  {cat}: {count}")
        print("Test set:")
        if 'None' in test_stats:
            print(f"  None: {test_stats['None']}")
        if 'Grammar' in test_stats:
            print(f"  Grammar: {test_stats['Grammar']}")
        if 'Mouth' in test_stats:
            print(f"  Mouth: {test_stats['Mouth']}")
        for cat, count in test_stats.items():
            if ' - ' in cat:  # Individual category counts
                print(f"  {cat}: {count}")
    
    return combined_folds

def test_model(model, test_loader, device):
    """Test the model and return all predictions."""
    model.eval()
    test_correct = 0
    test_total = 0
    predictions = []
    true_labels = []
    filenames = []  # Store filenames
    
    with torch.no_grad():
        for i, (input_arr_raw, target, filename) in enumerate(test_loader):
            input_arr = input_arr_raw[0][:,0:4,:,:]  # Hardcoded channel slice from original training
            input_arr = Tensor(input_arr).to(device)
            labels = torch.tensor([label_dic[x] for x in target], dtype=torch.long).to(device)
            
            outputs = model(input_arr)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            filenames.extend(filename)  # Store filenames

    test_acc = 100 * test_correct / test_total
    
    # Create results DataFrame with all predictions
    results = []
    for filename, true_label, pred_label in zip(filenames, true_labels, predictions):
        # Extract sign name from filename (e.g., "black(oo)" from "acoustic_diff_black(oo).npy")
        sign_name = filename.split('_')[-1].split('.')[0]  # Get last part before .npy
        results.append({
            'Sign': sign_name,
            'Truth': label_dic_reverse[int(true_label)],
            'Predicted': label_dic_reverse[int(pred_label)]
        })
    
    results_df = pd.DataFrame(results)
    return test_acc, results_df, predictions, true_labels

def save_cm_figure(true_label, predict_label, save_path, acc): 
    """Save confusion matrix figure."""
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
    plt.title(f"Confusion Matrix - Accuracy: {acc:.2f}%")
    
    # Save with high quality
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Base paths
    base_path = "/home/as4288/asl_acoustic_data_model"  # Update this to your base path
    experiment_path = os.path.join(base_path, "experiments/data/sign_mouth_combos_poi_300_360_th_50ch4_fusion_withcsvs/")
    save_path = os.path.join(base_path, "experiments/data/sign_mouth_combos_poi_300_360_th_50ch4_fusion_withcsvs/reloading")
    data_path = "/data/sign_mouth_combos"
    
    # Create folds
    print("Creating stratified folds...")
    folds = create_stratified_folds(data_path)
    
    # Process each fold (only 1-5)
    all_results = []
    # for fold_idx, (train_data, test_data) in enumerate(folds[:5]):  # Only process first 5 folds
    #     print(f"\nProcessing fold {fold_idx + 1}/5")  # Updated to show 5 instead of 6
    for fold_idx, (train_data, test_data) in enumerate(folds):
        # if fold_idx != 5:
        #     continue  # Skip all but the 6th fold (index 5)    
        # Create test loader
        test_dataset = CNNDataset(test_data, is_train=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=5,  # Same as original training
            shuffle=False,
            num_workers=0,
            collate_fn=collate_various_size
        )
        
        # Load the best model for this fold
        model = models.resnet18(num_classes=len(lst))
        model.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Same as original training
        
        checkpoint_path = os.path.join(experiment_path, f"fold{fold_idx+1}_best_checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ No checkpoint found at {checkpoint_path}, skipping fold")
            continue
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        
        # Test the model
        print(f"Testing fold {fold_idx + 1} model...")
        test_acc, results_df, predictions, true_labels = test_model(model, test_loader, device)
        
        # Add fold column
        results_df['Fold'] = f'Fold_{fold_idx + 1}'
        
        # Save results
        results_df.to_csv(os.path.join(save_path, f"test_results_fold{fold_idx+1}_full.csv"), index=False)
        
        # Save confusion matrix
        save_cm_figure(
            true_labels,
            predictions,
            os.path.join(save_path, f"confusion_matrix_fold{fold_idx+1}_full.png"),
            test_acc
        )
        
        print(f"Fold {fold_idx + 1} - Test Accuracy: {test_acc:.2f}%")
        print(f"Results saved to: {os.path.join(save_path, f'test_results_fold{fold_idx+1}_full.csv')}")
        
        # Store results for combined analysis
        all_results.append({
            'fold': fold_idx + 1,
            'accuracy': test_acc,
            'predictions': predictions,
            'true_labels': true_labels,
            'results_df': results_df
        })
    
    # Create combined results
    if all_results:
        print("\nCreating combined results...")
        
        # Combine all predictions
        all_predictions = []
        all_true_labels = []
        all_dfs = []
        total_acc = 0
        
        for result in all_results:
            all_predictions.extend(result['predictions'])
            all_true_labels.extend(result['true_labels'])
            all_dfs.append(result['results_df'])
            total_acc += result['accuracy']
        
        # Calculate average accuracy
        avg_acc = total_acc / len(all_results)
        
        # Save combined confusion matrix
        save_cm_figure(
            all_true_labels,
            all_predictions,
            os.path.join(save_path, "confusion_matrix_combined_full.png"),
            avg_acc
        )
        
        # Save combined results CSV
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(os.path.join(save_path, "test_results_combined_full.csv"), index=False)
        
        print(f"\nAverage accuracy across folds 1-5: {avg_acc:.2f}%")
        print(f"Combined results saved to: {os.path.join(save_path, 'test_results_combined_full.csv')}")
    else:
        print("⚠️ No results to combine!")

if __name__ == "__main__":
    main() 