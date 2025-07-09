import xml.etree.ElementTree as ET
import csv
import os
import glob
import re
from collections import defaultdict, Counter
# from nrclex import NRCLex
from collections import Counter
import nltk
from nltk.corpus import wordnet as wn_nltk
from nltk.corpus import sentiwordnet as swn
from collections import defaultdict
import wn
import torch
from transformers import pipeline

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

# Define the eyebrow positions we want to track
eyebrow_positions = [
    'raised',
    'lowered',
    'slightly lowered',
    'slightly raised',
    'raised-furrowed',
    'further raised',
    'further lowered',
    'right raised/left furrowed',
    'left raised/right furrowed',
    'left raised/right lowered',
    'further raised-furrowed'
]

# Create a nested defaultdict to store counts for eyebrow co-occurrences
gloss_counts = defaultdict(lambda: {'count': 0})
for pos in eyebrow_positions:
    for gloss in gloss_counts:
        gloss_counts[gloss][pos] = 0

def frames_overlap(start1, end1, start2, end2):
    """Check if two frame ranges overlap."""
    return not (end1 < start2 or start1 > end2)

def get_overlapping_signs(eyebrow_start, eyebrow_end, signs):
    """Get all signs that overlap with the given eyebrow movement."""
    overlapping = []
    for sign in signs:
        sign_start = int(sign.find('DOMINANT_HAND').get('START_FRAME', 0))
        sign_end = int(sign.find('DOMINANT_HAND').get('END_FRAME', 0))
        if frames_overlap(eyebrow_start, eyebrow_end, sign_start, sign_end):
            label = sign.find('LABEL').text.strip("'")
            overlapping.append(label)
    return overlapping
# Download required NLTK data
nltk.download('wordnet')
nltk.download('sentiwordnet')

# Initialize WN
wordnet = wn.Wordnet('oewn:2024')

class EmotionAnalyzer:
    def __init__(self):
        # Load NRC Emotion Lexicon, create dictionary of emotions to words
        self.emotion_lexicon = defaultdict(set)
        with open('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r', encoding='utf-8') as f:
            for line in f:
                word, emotion, score = line.strip().split('\t')
                if int(score) == 1:
                    self.emotion_lexicon[emotion].add(word)
        
        # Combine all emotional words into one flat set
        self.all_emotion_words = set()
        for words in self.emotion_lexicon.values():
            self.all_emotion_words |= words
    
    def get_emotion_words(self, text):
        """Get all emotion words in the text."""
        words = re.findall(r'\b\w+\b', text.lower())
        emotion_words = [w for w in words if w in self.all_emotion_words]
        return ';'.join(emotion_words)
    
    def get_sentence_emotions(self, text):
        """Get the emotional content of a sentence using NRC Emotion Lexicon."""
        if not text or not isinstance(text, str):
            return {}
        
        # Get words from text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Count emotions
        emotion_counts = defaultdict(int)
        for word in words:
            # Check which emotions this word is associated with
            for emotion, word_set in self.emotion_lexicon.items():
                if word in word_set:
                    emotion_counts[emotion] += 1
        
        return emotion_counts
    
    def get_dominant_emotions(self, emotion_counts):
        """Get the emotion(s) with the highest count."""
        if not emotion_counts:
            return ''
        
        max_count = max(emotion_counts.values())
        if max_count == 0:
            return ''
        
        dominant_emotions = [emotion for emotion, count in emotion_counts.items() if count == max_count]
        dominant_emotions.sort()  # sort alphabetically
        
        return ';'.join(dominant_emotions)
    
    def get_nltk_emotion(self, word):
        """Get emotion information using NLTK's WordNet/SentiWordNet."""
        synsets = wn_nltk.synsets(word)
        if not synsets:
            return None
        
        # Get average sentiment scores across all synsets
        pos_score = neg_score = obj_score = 0.0
        count = 0
        
        # get all possible meanings of a word and get sentiment scores for each meaning
        for synset in synsets: # synset: synonym set
            try:
                if hasattr(synset, 'name') and callable(synset.name):
                    senti_synset = swn.senti_synset(synset.name())
                    if senti_synset:
                        pos_score += senti_synset.pos_score()
                        neg_score += senti_synset.neg_score()
                        obj_score += senti_synset.obj_score()
                        count += 1
            except Exception as e:
                print(f"Error in NLTK emotion processing for word '{word}': {e}")
                continue
        
        if count > 0:
            pos_score /= count
            neg_score /= count
            obj_score /= count
            
            # Only return sentiment if the word is not primarily objective
            if obj_score < max(pos_score, neg_score):
                return {
                    'positive': pos_score,
                    'negative': neg_score,
                    'objective': obj_score
                }
        return None
    
    def get_wn_emotion(self, word):
        """Get emotion information using WN package."""
        try:
            synsets = wordnet.synsets(word)
            emotions = []
            
            for synset in synsets:
                try:
                    # Check hypernyms for emotion-related concepts
                    for hyper in synset.hypernyms(): #checking related words for emotions
                        try:
                            hyper_def = hyper.definition()
                            if isinstance(hyper_def, str) and ('emotion' in hyper_def or 'feeling' in hyper_def):
                                for lemma in hyper.lemmas():
                                    if hasattr(lemma, 'lemma') and callable(getattr(lemma, 'lemma')):
                                        lemma_name = lemma.lemma()
                                        if isinstance(lemma_name, str):
                                            emotions.append(lemma_name)
                        except Exception as e:
                            print(f"Error processing hypernym for word '{word}': {e}")
                            continue
                    
                    # Check definition for emotion words
                    synset_def = synset.definition()
                    if isinstance(synset_def, str):
                        for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']:
                            if emotion in synset_def.lower():
                                emotions.append(emotion)
                except Exception as e:
                    print(f"Error processing synset for word '{word}': {e}")
                    continue
            
            return list(set(emotions)) if emotions else None
        except Exception as e:
            print(f"Error in WN emotion processing for word '{word}': {e}")
            return None
    
    def analyze_text(self, text):
        """Analyze emotions in text using both WordNet implementations."""
        if not text or not isinstance(text, str):
            return {}
        
        words = re.findall(r'\b\w+\b', text.lower())
        analysis = {
            'nrc_pairs': [],
            'nltk_pairs': [],
            'wn_pairs': []
        }
        
        for word in words:
            # Get NRC emotions
            for emotion in self.emotion_lexicon:
                if word in self.emotion_lexicon[emotion]:
                    analysis['nrc_pairs'].append(f"{word}/{emotion}")
            
            # Get NLTK WordNet emotions - only include if not objective
            nltk_scores = self.get_nltk_emotion(word)
            if nltk_scores:
                # Get the dominant non-objective sentiment
                sentiments = [('positive', nltk_scores['positive']), 
                            ('negative', nltk_scores['negative'])]
                dominant = max(sentiments, key=lambda x: x[1])
                if dominant[1] > 0:  # Only include if sentiment strength > 0
                    analysis['nltk_pairs'].append(f"{word}/{dominant[0]}")
            
            # Get WN emotions
            wn_emotions = self.get_wn_emotion(word)
            if wn_emotions:
                for emotion in wn_emotions:
                    analysis['wn_pairs'].append(f"{word}/{emotion}")
        
        return analysis

# Initialize the emotion analyzer
emotion_analyzer = EmotionAnalyzer()

def get_emotion_pairs(text):
    """Get emotion word pairs from all three sources plus transformer emotions."""
    if not text or not isinstance(text, str):
        return '', '', '', ''
    
    analysis = emotion_analyzer.analyze_text(text)
    transformer_emotions = get_transformer_emotions(text)
    
    return (
        ';'.join(analysis['nrc_pairs']),
        ';'.join(analysis['nltk_pairs']),
        ';'.join(analysis['wn_pairs']),
        transformer_emotions
    )


# Initialize emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def get_transformer_emotions(text):
    """Get emotions from text using HuggingFace transformers."""
    if not text or not isinstance(text, str):
        return ''
    try:
        # Get emotion predictions
        emotions = emotion_classifier(text)[0]
        # Sort by score and get top emotions (those with score > 0.3)
        significant_emotions = [
            emotion['label'] 
            for emotion in sorted(emotions, key=lambda x: x['score'], reverse=True)
            if emotion['score'] > 0.3
        ]
        return ';'.join(significant_emotions) if significant_emotions else 'neutral'
    except Exception as e:
        print(f"Error in transformer emotion processing: {e}")
        return ''


# Get all XML files in the xml_files directory
xml_files = glob.glob('xml_files/*.xml')

if not xml_files:
    print("No XML files found in xml_files/ directory.")
    exit()

csv_file = 'xml_csvs/emotion_asl.csv'
word_count_file = 'xml_csvs/english_word_counts.csv'

# Initialize counters
word_counter = Counter()  # for English words
asl_counter = Counter()   # for ASL gloss words
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
    # Add separate columns for each emotion analysis method
    face_features = ['face_eye_brows', 'face_eye_gaze', 'face_eye_aperture', 'face_nose', 'face_mouth', 'face_cheeks']
    face_count_columns = [f'count_{feature}' for feature in face_features]
    header = ['#', 'utterance_id', 'translation', 'nrc_emotions', 'nltk_emotions', 'wn_emotions', 'transformer_emotions', 'asl_gloss', 'count_asl_gloss'] + list(feature_map.keys()) + face_count_columns
    writer.writerow(header)

    for xml_file in xml_files:
        print(f"Processing {xml_file}...")
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            collection_id = ''
            collection_elem = root.find('.//COLLECTION')
            if collection_elem is not None:
                collection_id = collection_elem.get('ID', '').strip("'")

            for utterance in root.findall('.//UTTERANCE'):
                utterance_id = utterance.get('ID', '').strip("'")
                translation_elem = utterance.find('TRANSLATION')
                translation = ''
                nrc_pairs = ''
                nltk_pairs = ''
                wn_pairs = ''
                transformer_emotions = ''
                if translation_elem is not None and translation_elem.text is not None:
                    translation = translation_elem.text.strip("'")
                    nrc_pairs, nltk_pairs, wn_pairs, transformer_emotions = get_emotion_pairs(translation)
                    
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
                            # Clean the ASL gloss
                            label_text = label_elem.text.strip("'")
                            cleaned_label = re.sub(r'[#+"]', '', label_text)
                            cleaned_label = re.sub(r'\(1h\)', '', cleaned_label)
                            cleaned_label = re.sub(r'\(2h\)', '', cleaned_label)
                            cleaned_label = cleaned_label.strip()
                            if cleaned_label:
                                labels.append(cleaned_label)
                                # Count ASL gloss words
                                asl_counter[cleaned_label] += 1

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
                
                # Add emotion pairs to row_data
                row_data = [collection_id, utterance_id, translation, nrc_pairs, nltk_pairs, wn_pairs, transformer_emotions, ';'.join(labels), asl_gloss_count] + feature_values + face_counts
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

# ASL gloss word counts
asl_count_file = 'xml_csvs/asl_gloss_word_counts.csv'
asl_counts = asl_counter.most_common()
with open(asl_count_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['asl_gloss', 'count'])
    for word, count in asl_counts:
        writer.writerow([word, count])

# Face feature word counts
for feature, counter in face_counters.items():
    feature_file = f'xml_csvs/{feature}_word_counts.csv'
    feature_counts = counter.most_common()
    with open(feature_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'count'])
        for word, count in feature_counts:
            writer.writerow([word, count])

# Face feature word counts (combined)
face_file = 'xml_csvs/face_word_counts.csv'
face_counts = face_counter.most_common()
with open(face_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'count'])
    for word, count in face_counts:
        writer.writerow([word, count])

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


# Process all XML files again for eyebrow-gloss analysis
for xml_file in xml_files:
    print(f"Processing {xml_file} for eyebrow analysis...")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for utterance in root.findall('.//UTTERANCE'):
            # Get all manual signs with their frame ranges
            manuals = utterance.find('MANUALS')
            if manuals is None:
                continue

            # Get all signs and their frame ranges
            signs = []
            for sign in manuals.findall('SIGN'):
                dom_hand = sign.find('DOMINANT_HAND')
                if dom_hand is not None and sign.find('LABEL') is not None:
                    label = sign.find('LABEL').text.strip("'")
                    # Clean the label
                    label = re.sub(r'[#+"]', '', label)
                    label = re.sub(r'\(1h\)', '', label)
                    label = re.sub(r'\(2h\)', '', label)
                    label = label.strip()
                    
                    if label:
                        start_frame = int(dom_hand.get('START_FRAME', 0))
                        end_frame = int(dom_hand.get('END_FRAME', 0))
                        signs.append((label, start_frame, end_frame))
                        # Increment the total count for this gloss
                        gloss_counts[label]['count'] += 1
                        # Initialize eyebrow counts if not already done
                        for pos in eyebrow_positions:
                            if pos not in gloss_counts[label]:
                                gloss_counts[label][pos] = 0

            # Get all eyebrow movements
            nonmanuals = utterance.find('NON_MANUALS')
            if nonmanuals is None:
                continue

            # Check each eyebrow movement
            for nm in nonmanuals.findall('NON_MANUAL'):
                label_elem = nm.find('LABEL')
                value_elem = nm.find('VALUE')
                
                if (label_elem is not None and value_elem is not None and 
                    label_elem.text and value_elem.text and
                    label_elem.text.strip("'") == 'eye brows'):
                    
                    eyebrow_value = value_elem.text.strip("'")
                    if eyebrow_value in eyebrow_positions:
                        start_frame = int(nm.get('START_FRAME', 0))
                        end_frame = int(nm.get('END_FRAME', 0))
                        
                        # Check which signs overlap with this eyebrow movement
                        for gloss, sign_start, sign_end in signs:
                            if frames_overlap(start_frame, end_frame, sign_start, sign_end):
                                gloss_counts[gloss][eyebrow_value] += 1

    except Exception as e:
        print(f"Error processing {xml_file} for eyebrow analysis: {e}")
        continue

# Write eyebrow-gloss results to CSV
output_file = 'xml_csvs/gloss_eyebrow_counts.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Write header
    header = ['asl_gloss', 'count'] + eyebrow_positions
    writer.writerow(header)
    
    # Sort glosses by count in descending order
    sorted_glosses = sorted(gloss_counts.keys(), key=lambda x: gloss_counts[x]['count'], reverse=True)
    
    # Write data for each gloss
    for gloss in sorted_glosses:
        row = [gloss, gloss_counts[gloss]['count']]
        row.extend(gloss_counts[gloss].get(pos, 0) for pos in eyebrow_positions)
        writer.writerow(row)

print(f"\nCreated eyebrow-gloss co-occurrence CSV file: {output_file}")
print(f"Total unique glosses analyzed: {len(gloss_counts)}")
