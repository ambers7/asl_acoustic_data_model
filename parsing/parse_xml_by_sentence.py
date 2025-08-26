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


def get_facial_expression_timing(nonmanuals, feature_label, utterance_start_frame):
    """Get facial expressions with their timing information."""
    expressions = []
    if nonmanuals is not None:
        for nm in nonmanuals.findall('NON_MANUAL'):
            label_elem = nm.find('LABEL')
            value_elem = nm.find('VALUE')
            if (label_elem is not None and value_elem is not None and 
                label_elem.text and value_elem.text):
                
                label_text = label_elem.text.strip("'")
                # Check if the label matches our feature label
                if label_text == feature_label:
                    # Get absolute frame numbers
                    start_attr = nm.get('START_FRAME')
                    end_attr = nm.get('END_FRAME')
                    print(f"Frame numbers - start: {start_attr}, end: {end_attr}")
                    
                    if start_attr is None or end_attr is None:
                        print(f"Warning: Missing frame numbers for {value_elem.text}")
                        continue
                        
                    start_frame = int(start_attr)
                    end_frame = int(end_attr)
                    
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

csv_file = 'parsing/xml_csvs/facial_expressions.csv'

with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ['#', 'utterance_id', 'translation', 'asl_gloss', 'count_asl_gloss', 
              'face_eye_brows', 'face_mouth', 'face_cheeks']
    writer.writerow(header)
    
    row_count = 0
    for xml_file in xml_files:
        print(f"\nProcessing {xml_file}...")
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for utterance in root.findall('.//UTTERANCE'):
                utterance_id = utterance.get('ID', '').strip("'")
                start_attr = utterance.get('START_FRAME')
                print(f"\nUtterance {utterance_id} - START_FRAME attribute: {start_attr}")
                
                if start_attr is None:
                    print(f"Warning: Missing START_FRAME for utterance {utterance_id}")
                    continue
                    
                utterance_start_frame = int(start_attr)
                
                # Get translation
                translation_elem = utterance.find('TRANSLATION')
                if translation_elem is None or translation_elem.text is None:
                    continue
                    
                translation = translation_elem.text.strip("'")
                print(f"Translation: {translation}")
                
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
                print("\nProcessing facial expressions:")
                print("Eyebrows:")
                eyebrows = get_facial_expression_timing(nonmanuals, 'eye brows', utterance_start_frame)
                print("\nMouth:")
                mouth = get_facial_expression_timing(nonmanuals, 'mouth', utterance_start_frame)
                print("\nCheeks:")
                cheeks = get_facial_expression_timing(nonmanuals, 'cheeks', utterance_start_frame)
                
                # Create a single row for the utterance
                row_count += 1
                row_data = [
                    row_count,
                    utterance_id,
                    translation,
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

print(f"\nCSV file '{csv_file}' created with {row_count} sentences.")