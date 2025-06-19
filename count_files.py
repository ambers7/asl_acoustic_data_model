import glob
import os

# Count files per letter for train directory
train_dir = '../train/'
train_files = glob.glob(os.path.join(train_dir, '*.npy'))
train_letter_counts = {}
for file_path in train_files:
    fname = os.path.basename(file_path)
    letter = fname.split('_')[-1][0]
    train_letter_counts[letter] = train_letter_counts.get(letter, 0) + 1

print("\nFiles per letter in train/ directory:")
for letter in sorted(train_letter_counts.keys()):
    print(f"Letter {letter}: {train_letter_counts[letter]} files")

# Count files per letter for test directory
test_dir = '../test/'
test_files = glob.glob(os.path.join(test_dir, '*.npy'))
test_letter_counts = {}
for file_path in test_files:
    fname = os.path.basename(file_path)
    letter = fname.split('_')[-1][0]
    test_letter_counts[letter] = test_letter_counts.get(letter, 0) + 1

print("\nFiles per letter in test/ directory:")
for letter in sorted(test_letter_counts.keys()):
    print(f"Letter {letter}: {test_letter_counts[letter]} files")

# Count files per letter for combined directory
all_data_dir = '../combined_data/'
all_files = glob.glob(os.path.join(all_data_dir, '*.npy'))

# Count files per letter
letter_counts = {}
for file_path in all_files:
    fname = os.path.basename(file_path)
    letter = fname.split('_')[-1][0]  # Gets the first character after the last underscore
    letter_counts[letter] = letter_counts.get(letter, 0) + 1

print("\nFiles per letter in combined_data/ directory:")
for letter in sorted(letter_counts.keys()):
    print(f"Letter {letter}: {letter_counts[letter]} files")