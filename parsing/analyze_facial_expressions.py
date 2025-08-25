import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('parsing/xml_csvs/emotion_asl.csv')

# Total number of sentences
total_sentences = len(df)

# Function to check if a row has facial expressions
def has_facial_expressions(row):
    face_columns = ['count_face_eye_brows', 'count_face_mouth', 'count_face_cheeks']
    return sum(row[face_columns]) > 0

# Function to count facial expressions in a row
def count_facial_expressions(row):
    face_columns = ['count_face_eye_brows', 'count_face_mouth', 'count_face_cheeks']
    return sum(row[face_columns])

# Function to check if a row has multiple facial expressions
def has_multiple_facial_expressions(row):
    face_columns = ['count_face_eye_brows', 'count_face_mouth', 'count_face_cheeks']
    return sum(row[col] > 0 for col in face_columns) > 1

# Function to check if facial expressions overlap
def has_overlapping_expressions(row):
    # If multiple expressions exist in the same sentence, they must overlap
    return has_multiple_facial_expressions(row)

# Calculate statistics
sentences_with_facial_exp = df[df.apply(has_facial_expressions, axis=1)]
sentences_with_multiple = df[df.apply(has_multiple_facial_expressions, axis=1)]
sentences_with_overlapping = df[df.apply(has_overlapping_expressions, axis=1)]

# Calculate percentages
percent_with_facial = (len(sentences_with_facial_exp) / total_sentences) * 100
percent_with_multiple = (len(sentences_with_multiple) / total_sentences) * 100
percent_overlapping = (len(sentences_with_overlapping) / total_sentences) * 100

# Calculate average number of facial expressions per sentence
avg_facial_expressions = df.apply(count_facial_expressions, axis=1).mean()

print(f"Total number of sentences analyzed: {total_sentences}")
print(f"Percentage of sentences with facial expressions: {percent_with_facial:.1f}%")
print(f"Percentage of sentences with multiple facial expressions: {percent_with_multiple:.1f}%")
print(f"Percentage of sentences with overlapping facial expressions: {percent_overlapping:.1f}%")
print(f"Average number of facial expressions per sentence: {avg_facial_expressions:.1f}")
