import pandas as pd
from collections import Counter
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('xml_csvs/asl_public_dataset.csv')
except FileNotFoundError:
    print("The file 'asl_public_dataset.csv' was not found. Please make sure it's in the correct directory.")
    exit()

# --- 1. Data Preparation ---

# Extract the 'asl_gloss' column which contains the glosses
glosses = df['asl_gloss'].dropna().tolist()
# print(glosses)

# Preprocess the glosses
preprocessed_glosses = []
for gloss in glosses:
    # Split into tokens (by semicolon and getting rid of whitespace)
    tokens = [token.strip() for token in gloss.split(';')]
    if tokens:
        # Add start and end markers to each gloss sequence
        preprocessed_glosses.append(['<s>'] + tokens + ['</s>'])

# --- 2. Counting N-grams ---

unigram_counts = Counter()
bigram_counts = Counter()

for sentence in preprocessed_glosses:
    # Count Unigrams
    unigram_counts.update(sentence)
    # Count Bigrams
    for i in range(len(sentence) - 1):
        bigram_counts[(sentence[i], sentence[i+1])] += 1

# Vocabulary size (V) is the number of unique unigrams
V = len(unigram_counts)

# --- 3. Calculating Probabilities with Smoothing ---

def calculate_add_one_smoothed_probability(word1, word2, unigram_counts, bigram_counts, V):
    """Calculates the add-one smoothed probability P(word2 | word1)"""
    bigram_key = (word1, word2)
    
    # Get the counts, adding 1 to the numerator
    numerator = bigram_counts[bigram_key] + 1
    
    # Get the counts, adding V to the denominator
    denominator = unigram_counts[word1] + V
    
    if denominator == 0:
        return 0.0 # Should not happen if V > 0
        
    return numerator / denominator

# --- 4. Example Usage ---

# Example: Calculate the probability of "BOOK" following "COLOR" with smoothing
prob_color_book_smoothed = calculate_add_one_smoothed_probability('COLOR', 'BOOK', unigram_counts, bigram_counts, V)

print("\n--- Results with Add-One Smoothing ---")
print(f"Vocabulary Size (V): {V}")
print(f"\nSmoothed Probability of 'BOOK' following 'COLOR', P(BOOK|COLOR): {prob_color_book_smoothed:.8f}")


# Example of calculating a sentence probability with smoothing
sentences_to_test = [
    "IX-1p POSS-1p GRANDMOTHER TEACH-1p ME ASL",
    "IX-1p LIKE BOOK",
    "POSS-1p MOTHER LOVE ME",
    "IX-1p GO SCHOOL",
    "TEACHER GIVE HOMEWORK",
    "IX-3p WANT EAT",
    "STUDENT LEARN SIGN LANGUAGE",
    "IX-1p NEED HELP",
    "FRIEND VISIT IX-1p"
]

summary_results = []
for sentence_to_test in sentences_to_test:
    tokens_to_test = ['<s>'] + sentence_to_test.split() + ['</s>']
    # Use log probabilities to avoid underflow with very small numbers
    sentence_log_probability_smoothed = 0.0

    print(f"\nCalculating smoothed probability for the sentence: '{sentence_to_test}'")
    for i in range(len(tokens_to_test) - 1):
        word1, word2 = tokens_to_test[i], tokens_to_test[i+1]
        prob = calculate_add_one_smoothed_probability(word1, word2, unigram_counts, bigram_counts, V)
        print(f"P({word2}|{word1}) = {prob:.8f}")
        sentence_log_probability_smoothed += np.log(prob)

    # The final probability is e^(sum of log probabilities)
    final_prob = np.exp(sentence_log_probability_smoothed)

    print(f"Smoothed Log Probability of the sentence is: {sentence_log_probability_smoothed:.4f}")
    print(f"Smoothed Probability of the sentence is: {final_prob:.2e}")
    summary_results.append((sentence_to_test, sentence_log_probability_smoothed))

# Print summary table
print("\n--- Summary: Sentence Log Probabilities ---")
for sent, log_prob in summary_results:
    print(f"Sentence: {sent}\nLog Probability: {log_prob:.4f}\n")