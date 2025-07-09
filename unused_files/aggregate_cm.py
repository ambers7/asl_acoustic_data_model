# aggregate confusion matrices from all runs

import pickle
from acoustic_model_resnet import save_cm_figure
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

all_ground_truth = []
all_predictions = []

classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z')
           
class_to_idx = {letter: idx for idx, letter in enumerate(classes)}
label_dic_reverse = {idx: letter for idx, letter in enumerate(classes)}  # Reverse mapping for confusion matrix


for run in range(0, 5): 
    with open(f'ground_truth_run{run}.pkl', 'rb') as f:
        all_ground_truth.extend(pickle.load(f))
    with open(f'predictions_run{run}.pkl', 'rb') as f:
        all_predictions.extend(pickle.load(f))

total = len(all_ground_truth)
correct = sum([1 for gt, pred in zip(all_ground_truth, all_predictions) if gt == pred])
accuracy = 100 * correct / total if total > 0 else 0

def save_cm_figure(true_label, predict_label, best_save_path, acc, classes): 
    true_labels = [label_dic_reverse[i] for i in true_label]
    predicted_labels = [label_dic_reverse[i] for i in predict_label]
    # Get unique class names and sort them (ensures correct label order)
    unique_classes = sorted(set(true_labels) | set(predicted_labels))
    # Compute confusion matrix with string labels
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    # Keep the label order in figure
    plt.xticks(ticks=np.arange(len(classes)) + 0.5, labels=classes, rotation=90)
    plt.yticks(ticks=np.arange(len(classes)) + 0.5, labels=classes, rotation=0)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Best Accuracy : %.3f"%acc + " %")
    plt.xticks(rotation=45)  # Rotate class labels for better visibility
    plt.yticks(rotation=0)
    plt.savefig(best_save_path, dpi=300, bbox_inches="tight")  # Saves as a high-quality PNG

save_cm_figure(
    all_ground_truth, 
    all_predictions, 
    'cms/acoustic_cnn_cm_runs0to2.png',
    accuracy, 
    classes
)
