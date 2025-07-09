import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

# Define the categories and their labels
lst = ["HAPPY", "SAD", "ANGRY", "TERRIFIED", "DISGUSTED", "SURPRISED",  # Emotions (6)
       "raise eyebrow", "furrowed eyebrow", "shake head side to side with lowered corners of the mouth and eyebrows",  # Grammar (3)
       "puffed", "oo", "mm", "CHA", "TH"]  # Mouth morphemes (5)

# Create label mappings
label_dic = {value: index for index, value in enumerate(lst)}
label_dic_reverse = {index: value for index, value in enumerate(lst)}

def save_cm_figure(true_label, predict_label, best_save_path, acc, lst): 
    # Convert numeric labels to class names
    true_labels = [label_dic_reverse[i] for i in true_label]
    predicted_labels = [label_dic_reverse[i] for i in predict_label]
    
    # Get unique class names and sort them to maintain order
    unique_classes = lst  # Use our predefined list to maintain category grouping
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized, 0)  # Replace NaN with 0
    
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

def analyze_performance_by_category(true_labels, predicted_labels):
    """Analyze model performance for each category."""
    category_ranges = {
        "Emotions": range(0, 6),
        "Grammar": range(6, 9),
        "Mouth Morphemes": range(9, 14)
    }
    
    category_metrics = {}
    for category_name, index_range in category_ranges.items():
        # Filter for this category
        category_mask = np.isin(true_labels, index_range)
        category_true = np.array(true_labels)[category_mask]
        category_pred = np.array(predicted_labels)[category_mask]
        
        # Calculate metrics
        correct = sum(1 for t, p in zip(category_true, category_pred) if t == p)
        total = len(category_true)
        accuracy = correct / total if total > 0 else 0
        
        category_metrics[category_name] = {
            "accuracy": accuracy,
            "samples": total,
            "correct": correct
        }
    
    return category_metrics

def load_all_test_results(base_path):
    """Load test results from all test sessions."""
    test_sessions = ['0301', '0601', '0901']  # Emotion, Grammar, Mouth sessions
    all_true_labels = []
    all_predicted_labels = []
    
    for session in test_sessions:
        session_path = os.path.join(base_path, f"session_{session}")
        if os.path.exists(session_path):
            results_file = os.path.join(session_path, "test_results.csv")
            if os.path.exists(results_file):
                results_df = pd.read_csv(results_file)
                all_true_labels.extend(results_df['True Label'].tolist())
                all_predicted_labels.extend(results_df['Predicted Label'].tolist())
    
    return all_true_labels, all_predicted_labels

def main():
    # Define paths
    experiment_path = "experiments/data/facial_expressions_poi_250_600_th_330ch1_fusion_facial_expressions"
    
    # Load test results from all sessions
    true_labels, predicted_labels = load_all_test_results(experiment_path)
    
    if not true_labels:  # If no results found in session folders, try the main results file
        results_file = os.path.join(experiment_path, "test_results.csv")
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            true_labels = results_df['True Label'].tolist()
            predicted_labels = results_df['Predicted Label'].tolist()
    
    if not true_labels:
        print("Error: No test results found!")
        return
    
    # Calculate overall accuracy
    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    total = len(true_labels)
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    # Generate confusion matrix
    save_cm_figure(
        true_labels,
        predicted_labels,
        experiment_path + "/",
        accuracy,
        lst
    )
    
    # Analyze performance by category
    metrics = analyze_performance_by_category(true_labels, predicted_labels)
    
    # Print results
    print("\nPerformance Analysis by Category:")
    print("="*50)
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    for category, stats in metrics.items():
        print(f"\n{category}:")
        print(f"  Accuracy: {stats['accuracy']*100:.2f}%")
        print(f"  Correct: {stats['correct']}/{stats['samples']} samples")

if __name__ == "__main__":
    main() 