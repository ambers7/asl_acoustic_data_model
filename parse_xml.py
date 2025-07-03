import xml.etree.ElementTree as ET
import csv
import os
import glob
import re
from collections import Counter
# from nrclex import NRCLex

# Map requested features to XML label names
feature_map = {
    'negative': 'negative',
    'wh_question': 'wh question',
    'yes_no_question': 'yes-no question',
    'topic_focus': 'topic/focus',
    'conditional_when': 'conditional/when',
    'role_shift': 'role shift',
    'head_pos_tilt_fr_bk': 'head pos: tilt fr/bk',
    'head_pos_turn': 'head pos: turn',
    'head_pose_tilt_side': 'head pos: tilt side',
    'head_pose_jut': 'head pos: jut',
    'head_mvmt_nod': 'head mvmt: nod',
    'head_mvmt_nod_cycles': 'head mvmt: nod cycles',
    'head_mvmt_shake': 'head mvmt: shake',
    'head_mvmt_side_to_side': 'head mvmt: side to side',
    'head_mvmt_jut': 'head mvmt: jut',
    'body_lean': 'body lean',
    'shoulders': 'shoulders',
    'face_eye_brows': 'eye brows',
    'face_eye_gaze': 'eye gaze',
    'face_eye_aperture': 'eye aperture',
    'face_nose': 'nose',
    'face_mouth': 'mouth',
    'face_cheeks': 'cheeks',
}

from collections import defaultdict
import re

# Load NRC Emotion Lexicon
emotion_lexicon = defaultdict(set)

with open('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r', encoding='utf-8') as f:
    for line in f:
        word, emotion, score = line.strip().split('\t')
        if int(score) == 1:
            emotion_lexicon[emotion].add(word)

# Optional: combine all emotional words into one flat set
all_emotion_words = set()
for words in emotion_lexicon.values():
    all_emotion_words |= words

def get_emotion_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    emotion_words = [w for w in words if w in all_emotion_words]
    return ';'.join(emotion_words)

# Get all XML files in the xml_files directory
xml_files = glob.glob('xml_files/*.xml')

if not xml_files:
    print("No XML files found in xml_files/ directory.")
    exit()

csv_file = 'xml_csvs/emotion_asl.csv'
word_count_file = 'xml_csvs/english_word_counts.csv'

# Initialize word counters
word_counter = Counter()
face_counters = {
    'face_eye_brows': Counter(),
    'face_eye_gaze': Counter(),
    'face_eye_aperture': Counter(),
    'face_nose': Counter(),
    'face_mouth': Counter(),
    'face_cheeks': Counter()
}
face_counter = Counter()  # Combined face counter
head_counter = Counter()

def clean_phrase(phrase):
    """Clean a phrase by trimming whitespace."""
    return phrase.strip()

with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Create header with count columns
    face_features = ['face_eye_brows', 'face_eye_gaze', 'face_eye_aperture', 'face_nose', 'face_mouth', 'face_cheeks']
    face_count_columns = [f'count_{feature}' for feature in face_features]
    header = ['#', 'utterance_id', 'translation', 'emotion_words', 'asl_gloss', 'count_asl_gloss'] + list(feature_map.keys()) + face_count_columns
    writer.writerow(header)

    for xml_file in xml_files:
        print(f"Processing {xml_file}...")
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get collection ID from the first COLLECTION element
            collection_id = ''
            collection_elem = root.find('.//COLLECTION')
            if collection_elem is not None:
                collection_id = collection_elem.get('ID', '').strip("'")

            for utterance in root.findall('.//UTTERANCE'):
                utterance_id = utterance.get('ID', '').strip("'")
                translation_elem = utterance.find('TRANSLATION')
                translation = ''
                emotion_words = ''
                if translation_elem is not None and translation_elem.text is not None:
                    # Get the English translation
                    translation = translation_elem.text.strip("'")
                    # Extract emotion words from the English translation
                    emotion_words = get_emotion_words(translation)
                    
                    # Count words in translation
                    if translation.strip():
                        words = translation.split()
                        for word in words:
                            cleaned_word = re.sub(r'[^\w\s]', '', word.lower())
                            cleaned_word = cleaned_word.strip()
                            if cleaned_word:  # Only count non-empty words
                                word_counter[cleaned_word] += 1

                labels = []
                manuals = utterance.find('MANUALS')
                if manuals is not None:
                    for sign in manuals.findall('SIGN'):
                        label_elem = sign.find('LABEL')
                        if label_elem is not None and label_elem.text:
                            # Clean the ASL gloss by removing #, +, (1h), (2h), and " characters
                            label_text = label_elem.text.strip("'")
                            # Remove the specified characters
                            cleaned_label = re.sub(r'[#+"]', '', label_text)  # Remove #, +, and "
                            cleaned_label = re.sub(r'\(1h\)', '', cleaned_label)  # Remove (1h)
                            cleaned_label = re.sub(r'\(2h\)', '', cleaned_label)  # Remove (2h)
                            # Clean up any extra whitespace
                            cleaned_label = cleaned_label.strip()
                            if cleaned_label:  # Only add non-empty labels
                                labels.append(cleaned_label)

                nonmanuals = utterance.find('NON_MANUALS')
                feature_values = []
                for feat, xml_label in feature_map.items():
                    values = []
                    if nonmanuals is not None:
                        for nm in nonmanuals.findall('NON_MANUAL'):
                            label_elem = nm.find('LABEL')
                            value_elem = nm.find('VALUE')
                            if label_elem is not None and value_elem is not None:
                                label_text = label_elem.text.strip("'") if label_elem.text else ''
                                if label_text == xml_label:
                                    values.append(value_elem.text.strip("'") if value_elem.text else '')
                    feature_values.append(';'.join(values))

                # Calculate counts
                asl_gloss_count = len(labels)
                
                # Calculate face feature counts and collect words
                face_counts = []
                for i, feature in enumerate(['face_eye_brows', 'face_eye_gaze', 'face_eye_aperture', 'face_nose', 'face_mouth', 'face_cheeks']):
                    # Find the index of this feature in feature_values
                    feature_index = list(feature_map.keys()).index(feature)
                    feature_value = feature_values[feature_index]
                    # Count non-empty values (split by semicolon)
                    if feature_value and feature_value.strip():
                        values = [x.strip() for x in feature_value.split(';') if x.strip()]
                        count = len(values)
                        # Count phrases for this face feature
                        for value in values:
                            cleaned_phrase = clean_phrase(value)
                            if cleaned_phrase:
                                face_counters[feature][cleaned_phrase] += 1
                                face_counter[cleaned_phrase] += 1  # Add to combined face counter
                    else:
                        count = 0
                    face_counts.append(count)
                
                # Collect head feature words
                head_features = ['head_pos_tilt_fr_bk', 'head_pos_turn', 'head_pose_tilt_side', 'head_pose_jut', 
                               'head_mvmt_nod', 'head_mvmt_nod_cycles', 'head_mvmt_shake', 'head_mvmt_side_to_side', 'head_mvmt_jut']
                for feature in head_features:
                    feature_index = list(feature_map.keys()).index(feature)
                    feature_value = feature_values[feature_index]
                    if feature_value and feature_value.strip():
                        values = [x.strip() for x in feature_value.split(';') if x.strip()]
                        for value in values:
                            cleaned_phrase = clean_phrase(value)
                            if cleaned_phrase:
                                head_counter[cleaned_phrase] += 1
                
                # Combine all data
                row_data = [collection_id, utterance_id, translation, emotion_words, ';'.join(labels), asl_gloss_count] + feature_values + face_counts
                writer.writerow(row_data)

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue

print(f"CSV file '{csv_file}' created.")

# Create word count CSVs
print(f"Creating word count CSVs...")

# English word counts
word_counts = word_counter.most_common()
with open(word_count_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'count'])
    for word, count in word_counts:
        writer.writerow([word, count])

print(f"English word count CSV '{word_count_file}' created.")
print(f"Total unique English words: {len(word_counts)}")
print(f"Total English word occurrences: {sum(count for _, count in word_counts)}")

# Face feature word counts
for feature, counter in face_counters.items():
    feature_file = f'xml_csvs/{feature}_word_counts.csv'
    feature_counts = counter.most_common()
    with open(feature_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'count'])
        for word, count in feature_counts:
            writer.writerow([word, count])
    print(f"{feature} word count CSV '{feature_file}' created.")
    print(f"  Total unique words: {len(feature_counts)}")
    print(f"  Total occurrences: {sum(count for _, count in feature_counts)}")

# Face feature word counts (combined)
face_file = 'xml_csvs/face_word_counts.csv'
face_counts = face_counter.most_common()
with open(face_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'count'])
    for word, count in face_counts:
        writer.writerow([word, count])

print(f"Face word count CSV '{face_file}' created.")
print(f"Total unique face words: {len(face_counts)}")
print(f"Total face word occurrences: {sum(count for _, count in face_counts)}")

# Head feature word counts (combined)
head_file = 'xml_csvs/head_word_counts.csv'
head_counts = head_counter.most_common()
with open(head_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'count'])
    for word, count in head_counts:
        writer.writerow([word, count])

print(f"Head word count CSV '{head_file}' created.")
print(f"Total unique head words: {len(head_counts)}")
print(f"Total head word occurrences: {sum(count for _, count in head_counts)}")

print("\nTop 10 most frequent English words:")
for rank, (word, count) in enumerate(word_counts[:10], 1):
    print(f"  {rank}. {word}: {count}") 