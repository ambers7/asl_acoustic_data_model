import xml.etree.ElementTree as ET
import csv
import os
import glob
import re
from collections import defaultdict

# Map requested features to XML label names
feature_map = {
    'face_eye_brows': 'eye brows',
    'face_mouth': 'mouth',
    'face_cheeks': 'cheeks'
}

def split_into_sentences(text):
    """Split text into sentences."""
    if not text:
        return []
    # Split on common sentence endings
    sentences = re.split(r'[.!?]+', text)
    # Remove empty sentences and strip whitespace
    return [s.strip() for s in sentences if s.strip()]

def get_facial_expression_timing(nonmanuals, feature_label):
    """Get facial expressions with their timing information."""
    expressions = []
    if nonmanuals is not None:
        for nm in nonmanuals.findall('NON_MANUAL'):
            label_elem = nm.find('LABEL')
            value_elem = nm.find('VALUE')
            if (label_elem is not None and value_elem is not None and 
                label_elem.text and value_elem.text and
                label_elem.text.strip("'") == feature_label):
                
                start_frame = int(nm.get('START_FRAME', 0))
                end_frame = int(nm.get('END_FRAME', 0))
                value = value_elem.text.strip("'")
                expressions.append({
                    'value': value,
                    'start_frame': start_frame,
                    'end_frame': end_frame
                })
    return expressions

def format_expressions_with_timing(expressions):
    """Format expressions list into string with timing information."""
    if not expressions:
        return ''
    return ';'.join([f"{exp['value']}({exp['start_frame']}-{exp['end_frame']})" 
                     for exp in expressions])

# Get all XML files in the xml_files directory
xml_files = glob.glob('parsing/xml_files/*.xml')

if not xml_files:
    print("No XML files found in xml_files/ directory.")
    exit()

csv_file = 'parsing/xml_csvs/facial_expressions_by_sentence.csv'

with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ['#', 'utterance_id', 'translation', 'asl_gloss', 'count_asl_gloss', 
              'face_eye_brows', 'face_mouth', 'face_cheeks']
    writer.writerow(header)
    
    row_count = 0
    for xml_file in xml_files:
        print(f"Processing {xml_file}...")
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for utterance in root.findall('.//UTTERANCE'):
                utterance_id = utterance.get('ID', '').strip("'")
                
                # Get translation and split into sentences
                translation_elem = utterance.find('TRANSLATION')
                if translation_elem is None or translation_elem.text is None:
                    continue
                    
                full_translation = translation_elem.text.strip("'")
                sentences = split_into_sentences(full_translation)
                
                # Get ASL gloss
                labels = []
                manuals = utterance.find('MANUALS')
                if manuals is not None:
                    for sign in manuals.findall('SIGN'):
                        label_elem = sign.find('LABEL')
                        if label_elem is not None and label_elem.text:
                            label_text = label_elem.text.strip("'")
                            cleaned_label = re.sub(r'[+"]', '', label_text)
                            cleaned_label = re.sub(r'\(1h\)', '', cleaned_label)
                            cleaned_label = re.sub(r'\(2h\)', '', cleaned_label)
                            cleaned_label = cleaned_label.strip()
                            if cleaned_label:
                                labels.append(cleaned_label)
                
                # Get facial expressions with timing
                nonmanuals = utterance.find('NON_MANUALS')
                eyebrows = get_facial_expression_timing(nonmanuals, 'eye brows')
                mouth = get_facial_expression_timing(nonmanuals, 'mouth')
                cheeks = get_facial_expression_timing(nonmanuals, 'cheeks')
                
                # Create a row for each sentence
                for sentence in sentences:
                    row_count += 1
                    row_data = [
                        row_count,
                        utterance_id,
                        sentence,
                        ';'.join(labels),
                        len(labels),
                        format_expressions_with_timing(eyebrows),
                        format_expressions_with_timing(mouth),
                        format_expressions_with_timing(cheeks)
                    ]
                    writer.writerow(row_data)

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue

print(f"CSV file '{csv_file}' created with {row_count} sentences.")
