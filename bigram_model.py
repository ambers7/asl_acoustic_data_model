import pandas as pd
import numpy as np
from collections import defaultdict
import math

def load_gloss_counts():
    """Load the gloss counts from the CSV file"""
    df = pd.read_csv('parsing/xml_csvs/gloss_eyebrow_counts.csv')
    return dict(zip(df['asl_gloss'], df['count']))

def load_sequences():
    """Load and process the ASL gloss sequences from the dataset"""
    df = pd.read_csv('parsing/xml_csvs/asl_public_dataset.csv')
    sequences = []
    
    for gloss_sequence in df['asl_gloss'].dropna():
        # Split by semicolon and clean up whitespace
        tokens = [token.strip() for token in gloss_sequence.split(';')]
        # Clean the tokens
        cleaned_tokens = []
        for token in tokens:
            # Remove (1h) and (2h) markers
            token = token.replace('(1h)', '').replace('(2h)', '')
            # Remove any resulting extra whitespace
            token = token.strip()
            if token:  # Only add non-empty tokens
                cleaned_tokens.append(token)
        
        if cleaned_tokens:
            # Add start and end markers
            sequences.append(['<s>'] + cleaned_tokens + ['</s>'])
    
    return sequences

def create_bigram_counts(sequences):
    """Create bigram counts from sequences of glosses"""
    bigram_counts = defaultdict(lambda: defaultdict(int))
    
    for sequence in sequences:
        for i in range(len(sequence) - 1):
            w1, w2 = sequence[i], sequence[i + 1]
            bigram_counts[w1][w2] += 1
    
    return bigram_counts

def calculate_bigram_probabilities(bigram_counts, unigram_counts, smoothing=1.0):
    """
    Calculate bigram probabilities with add-k smoothing
    P(w2|w1) = (count(w1,w2) + k) / (count(w1) + k*V)
    """
    bigram_probs = defaultdict(dict)
    vocab_size = len(unigram_counts)
    
    # Add start and end symbols to unigram counts if not present
    if '<s>' not in unigram_counts:
        unigram_counts['<s>'] = len([s for s in sequences if len(s) > 0])
    if '</s>' not in unigram_counts:
        unigram_counts['</s>'] = len([s for s in sequences if len(s) > 0])
    
    for w1 in unigram_counts:
        for w2 in unigram_counts:
            # Add-k smoothing
            numerator = bigram_counts[w1][w2] + smoothing
            denominator = unigram_counts[w1] + (smoothing * vocab_size)
            bigram_probs[w1][w2] = numerator / denominator
    
    return bigram_probs

def get_probability(bigram_probs, word1, word2):
    """Get the probability of word2 following word1"""
    if word1 not in bigram_probs:
        return 0.0
    return bigram_probs[word1].get(word2, 0.0)

def print_most_likely_next_words(bigram_probs, word, n=5):
    """Print the n most likely words to follow the given word"""
    if word not in bigram_probs:
        print(f"'{word}' not found in vocabulary")
        return
    
    # Sort by probability
    next_words = sorted(bigram_probs[word].items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {n} most likely words after '{word}':")
    for next_word, prob in next_words[:n]:
        if next_word != '</s>':  # Skip end symbol in display
            print(f"  {next_word}: {prob:.4f}")

def get_top_bigram_pairs(bigram_counts, unigram_counts, n=50, min_count=5):
    """
    Get the n most frequent bigram pairs with their probabilities
    Args:
        min_count: Minimum number of times a pair must occur to be included
    """
    pairs = []
    for w1 in bigram_counts:
        for w2 in bigram_counts[w1]:
            # Skip special tokens
            if w1 in ['<s>', '</s>'] or w2 in ['<s>', '</s>']:
                continue
            count = bigram_counts[w1][w2]
            # Only include pairs that occur more than min_count times
            if count < min_count:
                continue
            # Calculate probability using the total occurrences of w1
            total_w1 = sum(bigram_counts[w1].values())
            prob = count / total_w1 if total_w1 > 0 else 0
            # Calculate a combined score that considers both probability and frequency
            # This favors pairs that are both common and highly correlated
            score = prob * math.log(count + 1)  # log smooths the impact of frequency
            pairs.append((w1, w2, count, prob, score))
    
    # Sort by combined score
    pairs.sort(key=lambda x: x[4], reverse=True)
    return pairs[:n]

def main():
    # Load and process sequences
    print("\nLoading and processing ASL gloss sequences...")
    global sequences
    sequences = load_sequences()
    print(f"Loaded {len(sequences)} sequences")
    
    # Create bigram counts
    print("\nCreating bigram model...")
    bigram_counts = create_bigram_counts(sequences)
    
    # Calculate unigram counts from sequences
    unigram_counts = defaultdict(int)
    for sequence in sequences:
        for word in sequence:
            unigram_counts[word] += 1
    
    # Calculate probabilities
    print("Calculating bigram probabilities...")
    bigram_probs = calculate_bigram_probabilities(bigram_counts, unigram_counts)
    
    # Print top 50 most significant bigram pairs
    print("\nTop 50 ASL gloss transitions")
    print("(sorted by combination of probability and frequency, excluding those with under 5 occurences)")
    print("\nFormat: FIRST_WORD → SECOND_WORD")
    print("        Probability: X.XXX (occurred N times)")
    print("-" * 50)
    
    top_pairs = get_top_bigram_pairs(bigram_counts, unigram_counts, min_count=5)
    for w1, w2, count, prob, score in top_pairs:
        # Calculate what percentage of all sequences this pair represents
        percent_of_total = count / len(sequences) * 100
        print(f"{w1} → {w2}")
        print(f"Probability: {prob:.3f} (occurred {count} times, {percent_of_total:.1f}% of all sequences)")
        print("-" * 50)
    
    # Interactive mode
    while True:
        print("\nEnter two words to get their transition probability (or 'q' to quit)")
        word1 = input("First word: ").strip()
        if word1.lower() == 'q':
            break
        
        word2 = input("Second word: ").strip()
        if word2.lower() == 'q':
            break
        
        prob = get_probability(bigram_probs, word1, word2)
        if word1 in bigram_counts:
            count = bigram_counts[word1].get(word2, 0)
            total = sum(bigram_counts[word1].values())
            print(f"\nProbability of '{word2}' following '{word1}': {prob:.4f}")
            print(f"Occurred {count} times out of {total} appearances of '{word1}'")
        else:
            print(f"\n'{word1}' not found in dataset")
        
        # Also show what commonly follows the first word
        print_most_likely_next_words(bigram_probs, word1, n=5)

if __name__ == "__main__":
    main() 