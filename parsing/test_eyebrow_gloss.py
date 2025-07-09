import xml.etree.ElementTree as ET
import csv
import re
from collections import defaultdict

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

def frames_overlap(start1, end1, start2, end2):
    """Check if two frame ranges overlap."""
    return not (end1 < start2 or start1 > end2)

# Create a nested defaultdict to store counts for eyebrow co-occurrences
gloss_counts = defaultdict(lambda: {'count': 0})
for pos in eyebrow_positions:
    for gloss in gloss_counts:
        gloss_counts[gloss][pos] = 0

# Specify a single XML file to process
xml_file = 'xml_files/xml_extract_1-Ben-Introduction.xml'  # Change this to your test file

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
                            print(f"Found overlap: Gloss '{gloss}' with eyebrow '{eyebrow_value}' at frames {start_frame}-{end_frame}")

except Exception as e:
    print(f"Error processing {xml_file} for eyebrow analysis: {e}")

# Write eyebrow-gloss results to CSV
output_file = 'test_gloss_eyebrow_counts.csv'
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

# Print some sample results
print("\nSample results:")
for gloss in list(sorted_glosses)[:5]:  # Show first 5 glosses
    print(f"\nGloss: {gloss}")
    print(f"Total occurrences: {gloss_counts[gloss]['count']}")
    print("Eyebrow co-occurrences:")
    for pos in eyebrow_positions:
        count = gloss_counts[gloss].get(pos, 0)
        if count > 0:
            print(f"  {pos}: {count}") 