import xml.etree.ElementTree as ET
import glob
import csv
from collections import defaultdict, Counter
import re # Added for clean_gloss

# Define facial feature categories with complete list
facial_features = {
    'face_eye_brows': [
        'raised', 'lowered', 'slightly lowered', 'slightly raised', 'raised-furrowed',
        'further raised', 'further lowered', 'right raised/left furrowed',
        'left raised/right furrowed', 'left raised/right lowered', 'further raised-furrowed'
    ],
    'face_eye_gaze': [
        'left', 'right', 'up', 'down', 'down/left', 'down/right', 'into space', 'other',
        'up/right', 'up/left', 'to addressee', 'watch hands'
    ],
    'face_eye_aperture': [
        'blink', 'squint', 'wide', 'lowered lid', 'slightly squinted', 'further lowered',
        'slightly wide', 'further squinted', 'wider', 'intense'
    ],
    'face_nose': [
        'wrinkle', 'wrinkle left', 'wrinkle right', 'slightly wrinkled', 'slightly wrinkle'
    ],
    'face_mouth': [
        'closed', 'open', 'lips spread', 'lips pursed: mm', 'lips pursed: oo', 
        'lips spread & crnrs down', 'open & tongue visible', 'lips pursed: oo-tight',
        'open & round', 'bite lower lip', 'smile mouth open', 'tongue out',
        'open & tense', 'lips pursed corners down', 'open & corners down',
        'tongue on lwr lip', 'tongue sucked in quickly', 'tongue mvmt lateral',
        'raised upper lip'
    ],
    'face_cheeks': [
        'puffed', 'right tense', 'tensed', 'tense', 'puff right', 'puff left',
        'tensed right', 'tensed left', 'slightly tensed', 'further tensed',
        'less tensed', 'more tensed'
    ],
    'face_sounds': [
        'sh', 'brr', 'cs', 'cha', 'pow', 'puh', 'blow'
    ]
}

# Create reverse mapping for quick lookup
word_to_feature = {}
for feature, words in facial_features.items():
    for word in words:
        word_to_feature[word] = feature

# Initialize data structures
facial_expression_counts = defaultdict(int)  # Count of each facial expression
co_occurrence_counts = defaultdict(lambda: defaultdict(int))  # Facial expression -> Gloss -> Count

def frames_overlap(start1, end1, start2, end2):
    """Check if two frame ranges overlap."""
    return not (end1 < start2 or start1 > end2)

def clean_gloss(label):
    """Clean the ASL gloss label using the same rules as parse_xml.py."""
    if not label:
        return ''
    # Remove special characters and annotations
    label = re.sub(r'[+"]', '', label)  # Remove #, +, and quotes
    label = re.sub(r'\(1h\)', '', label)  # Remove (1h)
    label = re.sub(r'\(2h\)', '', label)  # Remove (2h)
    return label.strip()

def get_overlapping_signs(start_frame, end_frame, signs):
    """Get all signs that overlap with the given frame range."""
    overlapping = []
    for sign in signs:
        dom_hand = sign.find('DOMINANT_HAND')
        if dom_hand is not None:
            sign_start = int(dom_hand.get('START_FRAME', 0))
            sign_end = int(dom_hand.get('END_FRAME', 0))
            if frames_overlap(start_frame, end_frame, sign_start, sign_end):
                label = sign.find('LABEL')
                if label is not None and label.text:
                    # Clean the gloss label
                    clean_label = clean_gloss(label.text.strip("'"))
                    if clean_label:  # Only add non-empty labels
                        overlapping.append(clean_label)
    return overlapping

# Process XML files
xml_files = glob.glob('parsing/xml_files/*.xml')

for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for utterance in root.findall('.//UTTERANCE'):
            # Get all signs in this utterance
            manuals = utterance.find('MANUALS')
            if manuals is None:
                continue
                
            signs = manuals.findall('SIGN')
            
            # Process non-manual features
            nonmanuals = utterance.find('NON_MANUALS')
            if nonmanuals is None:
                continue
                
            for nm in nonmanuals.findall('NON_MANUAL'):
                label_elem = nm.find('LABEL')
                value_elem = nm.find('VALUE')
                
                if (label_elem is not None and value_elem is not None and 
                    label_elem.text and value_elem.text):
                    
                    value = value_elem.text.strip("'")
                    # Only count expressions that are in our categories
                    if value in word_to_feature:
                        facial_expression_counts[value] += 1
                        
                        # Get frame range for this expression
                        start_frame = int(nm.get('START_FRAME', 0))
                        end_frame = int(nm.get('END_FRAME', 0))
                        
                        # Find overlapping signs
                        overlapping_signs = get_overlapping_signs(start_frame, end_frame, signs)
                        
                        # Count co-occurrences
                        for sign in overlapping_signs:
                            co_occurrence_counts[value][sign] += 1

    except Exception as e:
        print(f"Error processing {xml_file}: {e}")
        continue

# Write results to CSV
output_file = 'facial_features_analysis_new.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Create header row
    header = ['facial_expression', 'feature_area', 'count']
    for i in range(1, 21):
        header.extend([f'top_{i}_word'])
    writer.writerow(header)
    
    # Write data rows
    for expression in sorted(facial_expression_counts.keys()):
        feature_area = word_to_feature[expression]  # We know this exists now
        count = facial_expression_counts[expression]
        
        # Get top 20 co-occurring words
        # Sort by count first, then alphabetically for ties
        top_words = sorted(co_occurrence_counts[expression].items(), 
                         key=lambda x: (-x[1], x[0]))[:20]
        
        # Pad with empty values if less than 20 co-occurrences
        while len(top_words) < 20:
            top_words.append(('', 0))
        
        # Create row
        row = [expression, feature_area, count]
        for word, count in top_words:
            row.append(f"{word} ({count})")
        
        writer.writerow(row)

print(f"\nCreated facial features analysis CSV file: {output_file}")
print(f"Total unique facial expressions analyzed: {len(facial_expression_counts)}")

# Print some sample results grouped by feature area
print("\nSample of facial expressions by feature area:")
for feature_area in facial_features.keys():
    print(f"\n{feature_area}:")
    expressions = [(exp, count) for exp, count in facial_expression_counts.items() 
                  if word_to_feature[exp] == feature_area]
    for expression, count in sorted(expressions, key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {expression}: {count}")
        # Print top 3 co-occurring words
        top_words = sorted(co_occurrence_counts[expression].items(), 
                         key=lambda x: (-x[1], x[0]))[:3]  # Sort by count first, then alphabetically
        if top_words:
            print("    Top co-occurring words:")
            for word, word_count in top_words:
                print(f"      - {word}: {word_count}")

print("\nFacial expression counts by feature area:")
for feature_area in facial_features.keys():
    count = sum(1 for exp in facial_expression_counts if word_to_feature[exp] == feature_area)
    total_occurrences = sum(facial_expression_counts[exp] 
                           for exp in facial_expression_counts 
                           if word_to_feature[exp] == feature_area)
    print(f"{feature_area}: {count} unique expressions, {total_occurrences} total occurrences") 