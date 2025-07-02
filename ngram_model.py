import pandas as pd
from collections import Counter

# Load the dataset
try:
    df = pd.read_csv('xml_csvs/asl_public_dataset.csv')
except FileNotFoundError:
    print("The file 'asl_public_dataset.csv' was not found. Please make sure it's in the correct directory.")
    exit()

# --- 1. Data Preparation ---

# Extract the 'asl_gloss' column which contains the glosses
glosses = df['asl_gloss'].dropna().tolist()
print(glosses)

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

# --- 3. Calculating Probabilities (for a bigram model) ---

def calculate_bigram_probability(word1, word2, unigram_counts, bigram_counts):
    """Calculates the unsmoothed probability P(word2 | word1)"""
    actual_bigram_key = (word1, word2)
    
    if unigram_counts[word1] == 0:
        return 0.0 # Cannot divide by zero
    
    return bigram_counts[actual_bigram_key] / unigram_counts[word1]


# --- 4. Example Usage ---

# Example: Calculate the probability of "BOOK" following "COLOR"
prob_color_book = calculate_bigram_probability('COLOR', 'BOOK', unigram_counts, bigram_counts)

print("\n--- Training Results ---")
print(f"Total unique unigrams (vocabulary size): {len(unigram_counts)}")
print(f"Total unique bigrams found: {len(bigram_counts)}")

print("\n--- Example Calculation ---")
print("Top 10 most common unigrams:", unigram_counts.most_common(10))
print("Top 10 most common bigrams:", bigram_counts.most_common(10))
print(f"\nProbability of 'BOOK' following 'COLOR', P(BOOK|COLOR): {prob_color_book:.4f}")

# Example of calculating a sentence probability
sentence_to_test = "IX-1p POSS-1p GRANDMOTHER TEACH-1p ME ASL"
tokens_to_test = ['<s>'] + sentence_to_test.split() + ['</s>']
sentence_probability = 1.0

print(f"\nCalculating probability for the sentence: '{sentence_to_test}'")
for i in range(len(tokens_to_test) - 1):
    word1, word2 = tokens_to_test[i], tokens_to_test[i+1]
    prob = calculate_bigram_probability(word1, word2, unigram_counts, bigram_counts)
    print(f"P({word2}|{word1}) = {prob:.4f}")
    if prob > 0:
        sentence_probability *= prob
    else:
        # If any part of the sequence has 0 probability, the whole sentence does (without smoothing)
        sentence_probability = 0
        break

print(f"\nUnsmoothed probability of the sentence is: {sentence_probability}")
print("\nNote: A probability of 0 for the sentence is common without smoothing, as it's likely that at least one bigram was not seen in the training data.")