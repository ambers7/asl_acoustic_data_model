import pandas as pd
from collections import Counter
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('xml_csvs/asl_public_dataset.csv')
except FileNotFoundError:
    print("The file 'asl_public_dataset.csv' was not found. Please make sure it's in the correct directory.")
    exit()

# --- 1. Create Paired Word-Facial Expression Tokens ---

# Define the columns for facial expressions
facial_expression_columns = [
    'negative', 'wh_question', 'yes_no_question', 'topic_focus', 'conditional_when',
    'face_eye_brows', 'face_eye_gaze', 'face_eye_aperture', 'face_nose', 'face_mouth', 'face_cheeks'
]

# Abbreviate column names for cleaner tokens
fe_abbreviations = {
    'negative': 'neg', 'wh_question': 'whq', 'yes_no_question': 'ynq', 'topic_focus': 'tf',
    'conditional_when': 'cond', 'face_eye_brows': 'eb', 'face_eye_gaze': 'eg',
    'face_eye_aperture': 'ea', 'face_nose': 'no', 'face_mouth': 'mo', 'face_cheeks': 'ch'
}


all_paired_token_sequences = []
# Use .itertuples() for efficient row iteration
for row in df.itertuples(index=False):
    # Create the facial expression string for the current row
    fe_parts = []
    for col in facial_expression_columns:
        value = getattr(row, col)
        # Ensure consistent formatting
        fe_parts.append(f"{fe_abbreviations[col]}-{value}")
    facial_expression_string = "_".join(fe_parts)
    
    # Get the gloss and split into words
    gloss = getattr(row, 'asl_gloss')
    if pd.notna(gloss):
        words = gloss.split()
        if not words:
            continue
            
        # Create the new sequence of paired tokens
        paired_sequence = ['<s>']
        for word in words:
            paired_token = f"{word}_{facial_expression_string}"
            paired_sequence.append(paired_token)
        paired_sequence.append('</s>')
        all_paired_token_sequences.append(paired_sequence)

# --- 2. Build the N-gram Model on Paired Tokens ---

unigram_counts = Counter()
bigram_counts = Counter()

for seq in all_paired_token_sequences:
    unigram_counts.update(seq)
    for i in range(len(seq) - 1):
        bigram_counts[(seq[i], seq[i+1])] += 1

# Vocabulary size (V) is now much larger
V = len(unigram_counts)

# --- 3. Calculate Smoothed Probabilities ---

def calculate_add_one_smoothed_probability(token1, token2, unigram_counts, bigram_counts, V):
    """Calculates smoothed probability for the paired tokens"""
    numerator = bigram_counts.get((token1, token2), 0) + 1
    denominator = unigram_counts.get(token1, 0) + V
    return numerator / denominator

# --- 4. Example Usage ---

print(f"--- Multimodal N-gram Model ---")
print(f"New vocabulary size (word-expression pairs): {V}")
print(f"Total unique bigrams of paired tokens: {len(bigram_counts)}\n")

# Example: What's the probability of 'NAME' with neutral expressions following 'MY' with neutral expressions?
# NOTE: This exact token would depend on the specific values in the CSV.
# This is a hypothetical example of what a token might look like.
neutral_face = "neg-0_whq-0_ynq-0_tf-0_cond-0_eb-0_eg-0_ea-0_no-0_mo-0_ch-0"
token1_example = f"MY_{neutral_face}"
token2_example = f"NAME_{neutral_face}"

prob_example = calculate_add_one_smoothed_probability(
    token1_example, token2_example, unigram_counts, bigram_counts, V
)

print(f"Example Smoothed Probability:")
print(f"P({token2_example[:15]}... | {token1_example[:15]}...) = {prob_example:.6e}")

print("\n--- Test Sentences with Their Actual Facial Expressions ---")
num_examples = 5  # Number of examples to test
for idx, row in enumerate(df.itertuples(index=False)):
    if idx >= num_examples:
        break
    # Build the facial expression string for this row
    fe_parts = []
    for col in facial_expression_columns:
        value = getattr(row, col)
        fe_parts.append(f"{fe_abbreviations[col]}-{value}")
    facial_expression_string = "_".join(fe_parts)
    
    gloss = getattr(row, 'asl_gloss')
    if pd.notna(gloss):
        words = gloss.split()
        if not words:
            continue
        tokens = ['<s>'] + [f"{word}_{facial_expression_string}" for word in words] + ['</s>']
        log_prob = 0.0
        for i in range(len(tokens) - 1):
            prob = calculate_add_one_smoothed_probability(tokens[i], tokens[i+1], unigram_counts, bigram_counts, V)
            log_prob += np.log(prob)
        print(f"Gloss: {gloss}")
        print(f"Facial Expression: {facial_expression_string}")
        print(f"Log Probability: {log_prob:.4f}\n")