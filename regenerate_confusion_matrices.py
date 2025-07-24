import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn as nn

# Define our categories
lst = [
    "raise eyebrows", "furrowed eyebrows", "shake head side to side with lowered corners of the mouth and eyebrows",  # Grammar (3)
    "puffed", "oo", "mm", "CHA", "TH"  # Mouth morphemes (5)
]

# Create label mappings
label_dic = {value: index for index, value in enumerate(lst)}
label_dic_reverse = {index: value for index, value in enumerate(lst)}

def save_cm_figure(true_label, predict_label, save_path, acc): 
    true_labels = [label_dic_reverse[i] for i in true_label]
    predicted_labels = [label_dic_reverse[i] for i in predict_label]
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=lst)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    
    plt.xticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=0)
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Accuracy: {acc:.2f}%")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# Copy over just the model architecture we need
class Imu1dImage2dModel(nn.Module):
    def __init__(self, num_classes=len(lst)):
        super(Imu1dImage2dModel, self).__init__()

        # 1D CNN for IMU data
        self.imu_cnn = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Pool to 1 to reduce to [batch, 128, 1]
        )

        # ResNet18 for Second Input Modality
        self.resnet2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.resnet2.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change input channels to 4
        self.resnet2.fc = nn.Identity()  # Remove final FC layer

        # Batch Normalization for Normalizing Feature Vectors
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(512)

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(128 + 512, num_classes)  # Combine IMU (128) + ResNet (512)

    def normalize(self, x):
        """Normalize the feature map to the range [0, 1]"""
        min_val = torch.min(x, dim=-1, keepdim=True)[0]
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val)
    
    def forward(self, x1, x2):
        # Extract Features
        B, C, H, W = x1.shape  # Input 1: 1D CNN for sequential IMU data
        x1 = x1.reshape(B, H, W)  # Height 3 - Channel, W - Time
        feat1 = self.imu_cnn(x1)
        feat1 = feat1.view(feat1.size(0), -1)   # Flatten to [batch, 128]

        feat2 = self.resnet2(x2)  # (B, 512)

        # Flatten Features
        feat1 = feat1.view(feat1.size(0), -1)  # (B, 128)
        feat2 = feat2.view(feat2.size(0), -1)  # (B, 512)

        # Normalize Features
        feat1 = self.bn1(feat1)  # Batch normalization
        feat2 = self.bn2(feat2)  # Batch normalization

        feat1 = self.normalize(feat1)  # (B, 128)
        feat2 = self.normalize(feat2)  # (B, 512)

        # Concatenate Features
        fused_features = torch.cat([feat1, feat2], dim=1)  # (B, 640)
        fused_features = self.normalize(fused_features)   # (B, 640)

        # Fully Connected Layer for Classification
        logits = self.fc(fused_features)  # (B, num_classes)
        return logits

# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_channel_slice = [0,1,2,3]

# Base directory where models are stored
base_dir = "/home/as4288/asl_acoustic_data_model/experiments/data/facial_expressions_poi_300_360_th_50ch4_fusion_poiwithoutemotions"

# Test cases
test_cases = [
    ['0601','0901'],
    ['1001'],
    ['1101'],
    ['1201'],
    ['1301']
]

all_predictions = []
all_true_labels = []

# Process each case
for case_idx, test_sessions in enumerate(test_cases, 1):
    print(f"\nProcessing Case {case_idx}")
    case_dir = os.path.join(base_dir, f"case_{case_idx}")
    model_path = os.path.join(case_dir, "best_model.pth")
    
    # Initialize model
    model = Imu1dImage2dModel(num_classes=len(lst))
    model.resnet2.conv1 = nn.Conv2d(len(input_channel_slice), 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Load saved model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load and process saved predictions if they exist
    predictions_file = os.path.join(case_dir, "predictions.npz")
    if os.path.exists(predictions_file):
        print(f"Loading saved predictions from {predictions_file}")
        data = np.load(predictions_file)
        predictions = data['predictions']
        true_labels = data['true_labels']
        accuracy = 100 * sum(1 for x, y in zip(predictions, true_labels) if x == y) / len(predictions)
    else:
        print(f"No saved predictions found at {predictions_file}")
        continue

    # Save confusion matrix
    save_cm_figure(
        true_labels,
        predictions,
        os.path.join(case_dir, "confusion_matrix_regenerated.png"),
        accuracy
    )
    print(f"Case {case_idx} - Test Accuracy: {accuracy:.2f}%")
    
    all_predictions.extend(predictions)
    all_true_labels.extend(true_labels)

# Generate overall confusion matrix
print("\nGenerating overall confusion matrix")
overall_acc = 100 * sum(1 for x, y in zip(all_predictions, all_true_labels) if x == y) / len(all_predictions)
save_cm_figure(
    all_true_labels,
    all_predictions,
    os.path.join(base_dir, "confusion_matrix_all_cases_regenerated.png"),
    overall_acc
)
print(f"Overall Accuracy: {overall_acc:.2f}%") 