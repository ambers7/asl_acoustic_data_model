import pandas as pd
import numpy as np
from collections import defaultdict
import math

def load_gloss_eyebrow_data():
    """Load the gloss and eyebrow data from CSV"""
    df = pd.read_csv('parsing/xml_csvs/gloss_eyebrow_counts.csv')
    # Get all eyebrow columns (excluding 'asl_gloss' and 'count')
    eyebrow_columns = [col for col in df.columns if col not in ['asl_gloss', 'count']]
    return df, eyebrow_columns

def create_gloss_eyebrow_counts(df, eyebrow_columns):
    """Create counts of each gloss-eyebrow combination"""
    gloss_eyebrow_counts = defaultdict(lambda: defaultdict(int))
    gloss_total_counts = defaultdict(int)
    
    for _, row in df.iterrows():
        gloss = row['asl_gloss']
        # The total count is from the count column
        total_count = row['count']
        gloss_total_counts[gloss] = total_count
        
        # For each eyebrow action, get its co-occurrence count
        for eyebrow in eyebrow_columns:
            if row[eyebrow] > 0:  # Only count non-zero values
                gloss_eyebrow_counts[gloss][eyebrow] = min(row[eyebrow], total_count)  # Cannot exceed total count
    
    return gloss_eyebrow_counts, gloss_total_counts

def calculate_probabilities(gloss_eyebrow_counts, gloss_total_counts):
    """Calculate probability of each eyebrow action given a gloss"""
    probabilities = defaultdict(dict)
    
    for gloss in gloss_eyebrow_counts:
        total = gloss_total_counts[gloss]
        if total > 0:  # Avoid division by zero
            for eyebrow, count in gloss_eyebrow_counts[gloss].items():
                # Probability cannot exceed 1.0
                probabilities[gloss][eyebrow] = min(count / total, 1.0)
    
    return probabilities

def get_top_pairs(probabilities, gloss_eyebrow_counts, gloss_total_counts, min_count=5):
    """Get the most significant gloss-eyebrow pairs"""
    pairs = []
    
    for gloss in probabilities:
        total_count = gloss_total_counts[gloss]
        if total_count >= min_count:  # Only include glosses that appear at least min_count times
            for eyebrow, prob in probabilities[gloss].items():
                count = gloss_eyebrow_counts[gloss][eyebrow]
                if count >= min_count:  # Only include pairs that co-occur at least min_count times
                    # Calculate a combined score that considers both probability and frequency
                    score = prob * math.log(count + 1)  # log smooths the impact of frequency
                    pairs.append((gloss, eyebrow, count, prob, score))
    
    # Sort by combined score
    pairs.sort(key=lambda x: x[4], reverse=True)
    return pairs

def main():
    print("\nLoading ASL gloss and eyebrow data...")
    df, eyebrow_columns = load_gloss_eyebrow_data()
    print(f"Found {len(eyebrow_columns)} eyebrow actions")
    
    print("\nCalculating gloss-eyebrow relationships...")
    gloss_eyebrow_counts, gloss_total_counts = create_gloss_eyebrow_counts(df, eyebrow_columns)
    probabilities = calculate_probabilities(gloss_eyebrow_counts, gloss_total_counts)
    
    # Print top pairs
    print("\nTop ASL gloss → eyebrow action transitions")
    print("(sorted by combination of probability and frequency, excluding glosses and co-occurrences with under 5 occurrences)")
    print("\nFormat: ASL_GLOSS → EYEBROW_ACTION")
    print("        Probability: X.XXX (co-occurred N times out of M total gloss occurrences)")
    print("-" * 50)
    
    top_pairs = get_top_pairs(probabilities, gloss_eyebrow_counts, gloss_total_counts, min_count=5)
    for gloss, eyebrow, count, prob, score in top_pairs[:50]:  # Show top 50 pairs
        total_gloss = gloss_total_counts[gloss]
        print(f"{gloss} → {eyebrow}")
        print(f"Probability: {prob:.3f} (co-occurred {count} times out of {total_gloss} total occurrences)")
        print("-" * 50)
    
    # Interactive mode
    while True:
        print("\nEnter an ASL gloss to see its eyebrow probabilities (or 'q' to quit)")
        gloss = input("ASL gloss: ").strip()
        if gloss.lower() == 'q':
            break
            
        if gloss in probabilities:
            print(f"\nEyebrow probabilities for '{gloss}':")
            # Sort by probability
            eyebrow_probs = sorted(probabilities[gloss].items(), key=lambda x: x[1], reverse=True)
            total_gloss = gloss_total_counts[gloss]
            print(f"This gloss appears {total_gloss} times in total")
            for eyebrow, prob in eyebrow_probs:
                count = gloss_eyebrow_counts[gloss][eyebrow]
                if count >= 5:  # Only show significant associations
                    print(f"  {eyebrow}: {prob:.3f} (co-occurred {count} times)")
        else:
            print(f"'{gloss}' not found in dataset")

if __name__ == "__main__":
    main() 