import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import logging
import random
from copy import deepcopy
from torch import Tensor
from math import ceil
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
import torch.nn as nn
import torch.optim as optim
import re
import torchvision.models as models

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

lst = [
    "black",
    "summer",
    "dontmind",
    "dontcare",
    "dry",
    "mother",
    "father",
    "nephew",
    "niece",
    "man",
    "woman",
    "understand",
    "sick",
    "disease",
    "red",
    "cute",
    "dorm",
    "home",
    "girl",
    "aunt",
    "twin",
    "restaurant",
    "see",
    "watch",
    "ant",
    "daily",
    "sunday",
    "wonderful",
    "star",
    "socks",
    "my",
    "I",
    "you",
    "your",
    "short",
    "child",
    "family",
    "class",
    "electric",
    "physics",
    "teach",
    "none",
    "sit",
    "chair",
    "nice",
    "clean",
    "train",
    "paper",
    "school",
    "read",
    "discuss",
    "late",
    "open",
    "run",
    "write",
    "carry",
    "sign",
    "drive",
    "bicycle",
    "study"
]

# Create label mappings
label_dic = {value: index for index, value in enumerate(lst)}
label_dic_reverse = {index: value for index, value in enumerate(lst)}
class_num = len(lst)

parser = argparse.ArgumentParser(description='Conditions')
parser.add_argument('--dataset_path', default='', type=str, help='dataset')
parser.add_argument('-poi','--point_of_interest', default='0,600', type=str, help='point of interest')
parser.add_argument('-g','--gpu_num', default=0, type=int, help='gpus')
parser.add_argument('--target_height', default=80, type=int, help='target')
parser.add_argument('--epoch', default=100, type=int, help='epoch')
parser.add_argument('--batch', default=5, type=int, help='batch')
parser.add_argument('--folder_name', default='', type=str, help='folder_name')

args = parser.parse_args()

def ensure_folder_exists(folder_path):
    """Check if a folder exists, and create it if not."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✅ Folder created: {folder_path}")
    else:
        print(f"📂 Folder already exists: {folder_path}")

# Setup parameters
num_epochs = args.epoch
target_height = args.target_height
gpu_set = args.gpu_num
dataset_folder = args.dataset_path
poi = args.point_of_interest
poi_list = poi.split(',')
folder_nm = args.folder_name
batch_size = args.batch

input_channel_slice = [0,1,2,3] #use channel 4 for acoustic data
input_channel = len(input_channel_slice)

folder = dataset_folder.split('/dataset/')[0]+'_poi_%s_%s'%(poi_list[0],poi_list[1])+'_th_%s'%(target_height)+'ch%s'%input_channel + '_3fold_%s'%folder_nm

best_save_path = "./experiments/%s/"%(folder)
ensure_folder_exists(best_save_path)

# Set up logging
def setup_logging(log_file_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, mode='a')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging(best_save_path + "logfile.txt")

def print_and_log(message):
    print(message)
    logger.info(message)

print_and_log("="*50)
print_and_log("Training script started")
print_and_log(f"Experiment folder: {best_save_path}")
print_and_log("="*50)

# Log the configuration
print_and_log("Model Configuration:")
print_and_log(f"Number of classes: {class_num}")
print_and_log("="*50)

def save_checkpoint(model, optimizer, epoch, best_acc=0.0, filename=best_save_path + "best_checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,  
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_accuracy": best_acc
    }
    torch.save(checkpoint, filename)
    print_and_log(f"✅ Checkpoint saved at epoch {epoch+1} with best accuracy: {best_acc:.2f}%")

def collate_various_size(batch):
    # Each batch item is (acoustic_data, label, filename)
    data_list_arr = [x[0] for x in batch]  # Get acoustic data directly
    target = [x[1] for x in batch]        # Get label
    filenames = [x[2] for x in batch]     # Get filename
    
    data_max_size = max([x.shape[1] for x in data_list_arr])
    
    window_size = 10
    target_length = data_max_size 
    target_length = ceil(target_length / window_size) * window_size
   
    # Create zero array with shape (batch_size, channels, max_length, height)
    data_arr = np.zeros((len(batch), data_list_arr[0].shape[0], target_length, data_list_arr[0].shape[2]))
    
    # Fill in the data with random horizontal shifting
    for i in range(len(data_list_arr)):
        start_x = random.randint(0, target_length - data_list_arr[i].shape[1])
        data_arr[i, :, start_x:start_x + data_list_arr[i].shape[1], :] = data_list_arr[i]

    # Swap axes to match expected input format
    data_arr = data_arr.swapaxes(2, 3)
        
    return data_arr, target, filenames

class DataSplitter:
    def __init__(self, train_data, test_data, BATCH_SIZE, WORKER_NUM):
        print_and_log('train length: ' + str(len(train_data)))
        print_and_log('test length: ' + str(len(test_data)))
        
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
            
        test_dataset = CNNDataset(test_data, is_train=False)
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
        input_arr = self.data[index][0]  # Get acoustic data
        output_arr = deepcopy(self.data[index][1])  # Get label
        filename = self.data[index][2]  # Get filename

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
            padded_input_fn = padded_input_fn[:,:,:target_height]

        return padded_input_fn, output_arr, filename

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

def read_from_folder(session_num, data_path, is_train=False):
    # Ensure session_num has 4 digits
    session_num = str(session_num).zfill(4)
    
    # Construct the full path correctly
    file_path = os.path.join(data_path + session_num)
    file_echo_diff = os.path.join(file_path, 'acoustic', 'diff')

    # Check if directory exists
    if not os.path.exists(file_echo_diff):
        print_and_log(f"Error: Directory not found: {file_echo_diff}")
        return []

    # Get sorted file list, excluding hidden files
    file_echo_diff_list = sorted([f for f in os.listdir(file_echo_diff) 
                                if not f.startswith('.') and f.endswith('.npy')])

    data_pairs = []
    n_bad = 0
    bad_signal_remove_length = 5
    
    # Initialize counters for each category
    category_counts = {label: 0 for label in lst}
    
    print_and_log(f"\nProcessing files in session {session_num}:")
    
    for file in file_echo_diff_list:
        try:
            # Extract label from filename (e.g., 'acoustic_diff_9_dorm(cs).npy')
            # label_match = file.split('(')[-1].split(')')[0].lower()
            # label_match = file.split('_')[-1].split('.')[0].split('(')[0]
            # label_match = re.search(r'_(\w+)\(', file).group(1)
            label_match = re.search(r'_([^_()]+)\(', file).group(1)
            
            # Skip if the label is not in our defined categories
            if label_match not in lst:
                print_and_log(f"Warning: Skipping unknown label '{label_match}' from file {file}")
                continue

            # Load acoustic data
            file_path = os.path.join(file_echo_diff, file)
            profiles = np.load(file_path, allow_pickle=True)
            profile_data_piece = profiles.copy()
            profile_data_piece = profile_data_piece.swapaxes(1, 2)

            if profile_data_piece.shape[1] > 50:  # check the data quality 
                data_pairs.append((
                    profile_data_piece[:,:-bad_signal_remove_length,:],
                    label_match,
                    file
                ))
                category_counts[label_match] += 1
            else:
                n_bad += 1
                print_and_log(f"Skipped due to quality check (length <= 50)")

        except Exception as e:
            print_and_log(f"Error processing file {file}: {str(e)}")
            n_bad += 1
            continue

    # Print category statistics
    print_and_log(f"\nCategory distribution for session {session_num}:")
    # print_and_log("-" * 40)
    # print_and_log("Grammar signs:")
    # for label in lst[1:4]:
    #     print_and_log(f"  {label}: {category_counts[label]}")
    # print_and_log("\nMouth morphemes:")
    # for label in lst[4:]:
    #     print_and_log(f"  {label}: {category_counts[label]}")
    print_and_log("\nNone category:")
    print_and_log(f"  none: {category_counts['none']}")
    print_and_log("-" * 40)
    
    if n_bad:
        print_and_log(f'     {n_bad} bad data pieces')

    return data_pairs

def save_cm_figure(true_label, predict_label, filepath, acc): 
    # Convert numeric labels to class names
    true_labels = [label_dic_reverse[i] for i in true_label]
    predicted_labels = [label_dic_reverse[i] for i in predict_label]
    
    # Create figure
    plt.figure(figsize=(20, 16))
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=lst)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=lst, yticklabels=lst)
    
    # Customize labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add titles and labels
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix\nAccuracy: {acc:.2f}%")
    
    # Save with high quality
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

# Define the three folds
folds = [
    {'test': ['0101'], 'train': ['0201', '0301']},
    {'test': ['0201'], 'train': ['0101', '0301']},
    {'test': ['0301'], 'train': ['0101', '0201']}
]

# Dictionary to store results for each fold
fold_results = {}

# Train on each fold
for fold_idx, fold in enumerate(folds):
    print_and_log("="*50)
    print_and_log(f"Training on fold {fold_idx + 1}/3")
    print_and_log(f"Test session: {fold['test'][0]}")
    print_and_log(f"Train sessions: {', '.join(fold['train'])}")
    
    # Load training data
    train_data = []
    for session in fold['train']:
        train_data.extend(read_from_folder(session, dataset_folder + '/dataset/session_'))
    
    # Load test data
    test_data = []
    for session in fold['test']:
        test_data.extend(read_from_folder(session, dataset_folder + '/dataset/session_'))
    
    # Create data loaders
    data_splitter = DataSplitter(train_data, test_data, batch_size, 0)
    train_loader = data_splitter.train_loader
    test_loader = data_splitter.test_loader
    
    # Initialize model for this fold
    model = models.resnet18(num_classes=class_num)
    model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
    device = torch.device(f"cuda:{gpu_set}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize tracking variables
    best_val_acc = 0.0
    fold_dir = os.path.join(best_save_path, f"fold_{fold_idx+1}")
    ensure_folder_exists(fold_dir)
    
    # Training loop for this fold
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training step
        for inputs, target, _ in train_loader:
            inputs = Tensor(inputs).to(device)
            labels = torch.tensor([label_dic[x] for x in target], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
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
            all_predictions = []
            all_true_labels = []
            all_filenames = []
            
            with torch.no_grad():
                for inputs, target, filenames in test_loader:
                    inputs = Tensor(inputs).to(device)
                    labels = torch.tensor([label_dic[x] for x in target], dtype=torch.long).to(device)
                    
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_true_labels.extend(labels.cpu().numpy())
                    all_filenames.extend(filenames)
            
            test_acc = 100 * test_correct / test_total
            
            if test_acc > best_val_acc:
                best_val_acc = test_acc
                
                # Save model checkpoint
                save_checkpoint(model, optimizer, epoch, best_acc=best_val_acc,
                             filename=os.path.join(fold_dir, "best_checkpoint.pth"))
                
                # Save confusion matrix
                save_cm_figure(all_true_labels, all_predictions,
                             os.path.join(fold_dir, "confusion_matrix.png"),
                             best_val_acc)
                
                # Save detailed results
                results = []
                for filename, true_label, pred_label in zip(all_filenames, all_true_labels, all_predictions):
                    results.append({
                        'File': filename,
                        'True Label': label_dic_reverse[true_label],
                        'Predicted Label': label_dic_reverse[pred_label],
                        'Correct': true_label == pred_label
                    })
                pd.DataFrame(results).to_csv(os.path.join(fold_dir, "results.csv"), index=False)
                
                # Store best results
                fold_results[fold_idx] = {
                    'test_session': fold['test'][0],
                    'train_sessions': fold['train'],
                    'accuracy': best_val_acc,
                    'predictions': all_predictions,
                    'true_labels': all_true_labels,
                    'filenames': all_filenames
                }
            
            print_and_log(f"Epoch [{epoch+1}/{num_epochs}], "
                         f"Loss: {running_loss/len(train_loader):.4f}, "
                         f"Train Acc: {100*correct/total:.2f}%, "
                         f"Test Acc: {test_acc:.2f}%, "
                         f"Best: {best_val_acc:.2f}%")
    
    # Clear GPU memory
    del model, optimizer
    torch.cuda.empty_cache()

# Print final summary
print_and_log("\n" + "="*50)
print_and_log("Training Complete - Summary of All Folds")
print_and_log("="*50)

total_acc = 0
for fold_idx, results in fold_results.items():
    print_and_log(f"Fold {fold_idx + 1}:")
    print_and_log(f"  Test Session: {results['test_session']}")
    print_and_log(f"  Train Sessions: {', '.join(results['train_sessions'])}")
    print_and_log(f"  Best Accuracy: {results['accuracy']:.2f}%")
    total_acc += results['accuracy']

print_and_log("-"*50)
print_and_log(f"Average Accuracy Across All Folds: {total_acc/3:.2f}%")
print_and_log("="*50)

# Create combined confusion matrix
all_predictions = []
all_true_labels = []
for results in fold_results.values():
    all_predictions.extend(results['predictions'])
    all_true_labels.extend(results['true_labels'])

save_cm_figure(all_true_labels, all_predictions,
               os.path.join(best_save_path, "confusion_matrix_combined.png"),
               total_acc/3)

# Save combined results
all_results = []
for fold_idx, results in fold_results.items():
    for filename, true_label, pred_label in zip(results['filenames'],
                                              results['true_labels'],
                                              results['predictions']):
        all_results.append({
            'File': filename,
            'True Label': label_dic_reverse[true_label],
            'Predicted Label': label_dic_reverse[pred_label],
            'Correct': true_label == pred_label,
            'Fold': f"Fold_{fold_idx+1}",
            'Test Session': results['test_session'],
            'Train Sessions': ', '.join(results['train_sessions'])
        })

pd.DataFrame(all_results).to_csv(os.path.join(best_save_path, "all_results.csv"), index=False)
print_and_log("\nSaved combined results to all_results.csv") 