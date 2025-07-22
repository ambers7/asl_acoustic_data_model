import os
import re
from collections import Counter

lst = {
    "black", "summer", "dontmind", "dontcare", "dry",
    "mother", "father", "nephew", "niece", "man",
    "woman", "understand", "sick", "disease", "red",
    "cute", "dorm", "home", "girl", "aunt",
    "twin", "restaurant", "see", "watch", "ant",
    "daily", "sunday", "wonderful", "star", "socks",
    "my", "I", "you", "your", "short",
    "child", "family", "class", "electric", "physics",
    "teach", "none", "sit", "chair", "nice",
    "clean", "train", "paper", "school", "read",
    "discuss", "late", "open", "run", "write",
    "carry", "sign", "drive", "bicycle", "study"
}
print(len(lst))


lst_1 = {
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
}
# List of session folders
session_folders = ["session_0101", "session_0201", "session_0301"]

# Base directory (modify if needed)
base_dir = "/data/sign_mouth_combos/dataset"  # or wherever your folders are located

# Pattern to extract base word before parenthesis
word_pattern = re.compile(r'^([^\(]+)')

for session in session_folders:
    path = os.path.join(base_dir, session, "gnd_truth.txt")
    
    if not os.path.exists(path):
        print(f"⚠️ gnd_truth.txt not found in {session}")
        continue

    unique_words = set()
    word_counts = Counter()


    with open(path, "r") as file:
        for line in file:
            parts = line.strip().split(';')
            if len(parts) >= 4:
                label = parts[3]  # e.g., family(shake)
                match = word_pattern.match(label)
                if match:
                    base_word = match.group(1)
                    unique_words.add(base_word)
                    word_counts[base_word] += 1

    print(f"📂 {session}: {len(unique_words)} unique words")
    print("   ➤", sorted(unique_words), "\n")

    missing_in_session = lst - unique_words
    extra_in_session = unique_words - lst

    print(f"  ✅ Missing from session (in reference but not in session):")
    print(f"     ➤ {sorted(missing_in_session) if missing_in_session else 'None'}")

    print(f"  ❌ Extra in session (not in reference list):")
    print(f"     ➤ {sorted(extra_in_session) if extra_in_session else 'None'}")

    repeated = {word: count for word, count in word_counts.items() if count > 10}
    if repeated:
        print(f"  🔁 Repeated words:")
        for word, count in repeated.items():
            print(f"     - {word}: {count} times")
    else:
        print("  ✅ No repeated words")

print(lst==lst_1)

lst_2 = [
    "black", "summer", "dontmind", "dontcare", "dry",
    "mother", "father", "nephew", "niece", "man",
    "woman", "understand", "sick", "disease", "red",
    "cute", "dorm", "home", "girl", "aunt",
    "twin", "restaurant", "see", "watch", "ant",
    "daily", "sunday", "wonderful", "star", "socks",
    "my", "I", "you", "your", "short",
    "child", "family", "class", "electric", "physics",
    "teach", "none", "sit", "chair", "nice",
    "clean", "train", "paper", "school", "read",
    "discuss", "late", "open", "run", "write",
    "carry", "sign", "drive", "bicycle", "study"
]   
print(len(lst_2))

asl_sign_locations = {
    "black": "upper face", "summer": "upper face", "dontmind": "lower face", "dontcare": "lower face", "dry": "lower face",
    "mother": "lower face", "father": "upper face", "nephew": "upper face", "niece": "lower face", "man": "upper face",
    "woman": "lower face", "understand": "upper face", "sick": "body", "disease": "body", "red": "lower face",
    "cute": "lower face", "dorm": "lower face", "home": "lower face", "girl": "lower face", "aunt": "lower face",
    "twin": "lower face", "restaurant": "lower face", "see": "upper face", "watch": "upper face", "ant": "body",
    "daily": "lower face", "sunday": "upper face", "wonderful": "upper face", "star": "upper face", "socks": "body",
    "my": "body", "I": "body", "you": "body", "your": "body", "short": "body", "child": "body", "family": "body",
    "class": "body", "electric": "body", "physics": "body", "teach": "upper face", "none": "body", "sit": "body",
    "chair": "body", "nice": "body", "clean": "body", "train": "body", "paper": "body", "school": "body",
    "read": "body", "discuss": "body", "late": "body", "open": "body", "run": "body", "write": "body",
    "carry": "body", "sign": "body", "drive": "body", "bicycle": "body", "study": "body"
}
print(len(asl_sign_locations))