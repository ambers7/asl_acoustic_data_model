import xml.etree.ElementTree as ET
import csv
import glob
import re
import os

def clean_gloss(text):
    """Clean the ASL gloss text."""
    if text is None:
        return ''
    # Remove special characters and clean up the text
    text = text.strip("'")
    text = re.sub(r'[#+"]', '', text)
    text = re.sub(r'\(1h\)', '', text)
    text = re.sub(r'\(2h\)', '', text)
    return text.strip()

def process_xml_files():
    # Get script directory and construct paths relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, 'xml_files', 'xml_extract_1-Ben-Introduction.xml')
    output_dir = os.path.join(script_dir, 'xml_csvs')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all XML files
    xml_files = glob.glob(xml_path)
    
    if not xml_files:
        print(f"No XML files found at path: {xml_path}")
        return
    
    # Create output CSV file
    output_file = os.path.join(output_dir, 'frame_utterance_map.csv')
    
    # Second pass: create frame-to-utterance mapping
    print("Creating frame-to-utterance mapping...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['#', 'utterance_id', 'frame', 'manual_signs', 'non_manual_signs'])
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                collection_id = ''
                collection_elem = root.find('.//COLLECTION')
                if collection_elem is not None:
                    collection_id = collection_elem.get('ID', '').strip("'")
                
                # Dictionary to store frame data
                frame_data = {}
                
                # Process each utterance
                for utterance in root.findall('.//UTTERANCE'):
                    utterance_id = utterance.get('ID', '').strip("'")
                    
                    # Process manual signs
                    manuals = utterance.find('MANUALS')
                    if manuals is not None:
                        for sign in manuals.findall('SIGN'):
                            dom_hand = sign.find('DOMINANT_HAND')
                            if dom_hand is not None:
                                start = int(dom_hand.get('START_FRAME', 0))
                                end = int(dom_hand.get('END_FRAME', 0))
                                label_elem = sign.find('LABEL')
                                if label_elem is not None:
                                    label = clean_gloss(label_elem.text)
                                    if label:
                                        for frame in range(start, end + 1):
                                            if frame not in frame_data:
                                                frame_data[frame] = {
                                                    'utterance_ids': set(),
                                                    'manual_signs': set(),
                                                    'non_manual_signs': set()
                                                }
                                            frame_data[frame]['utterance_ids'].add(utterance_id)
                                            frame_data[frame]['manual_signs'].add(label)
                    
                    # Process non-manual signs
                    nonmanuals = utterance.find('NON_MANUALS')
                    if nonmanuals is not None:
                        for nm in nonmanuals.findall('NON_MANUAL'):
                            start = int(nm.get('START_FRAME', 0))
                            end = int(nm.get('END_FRAME', 0))
                            label_elem = nm.find('LABEL')
                            value_elem = nm.find('VALUE')
                            if label_elem is not None and value_elem is not None:
                                label = clean_gloss(label_elem.text)
                                value = clean_gloss(value_elem.text)
                                if label and value:
                                    nm_text = f"{label}:{value}"
                                    for frame in range(start, end + 1):
                                        if frame not in frame_data:
                                            frame_data[frame] = {
                                                'utterance_ids': set(),
                                                'manual_signs': set(),
                                                'non_manual_signs': set()
                                            }
                                        frame_data[frame]['utterance_ids'].add(utterance_id)
                                        frame_data[frame]['non_manual_signs'].add(nm_text)
                
                # Write frame data in order
                if frame_data:
                    total_frames = max(frame_data.keys()) + 1
                    frames_with_utterances = len(frame_data)
                    
                    # Write frames in order
                    for frame in sorted(frame_data.keys()):
                        data = frame_data[frame]
                        writer.writerow([
                            collection_id,
                            ';'.join(sorted(data['utterance_ids'])),
                            frame,
                            ';'.join(sorted(data['manual_signs'])),
                            ';'.join(sorted(data['non_manual_signs']))
                        ])
                    
                    print(f"\nProcessed {total_frames} total frames")
                    print(f"Found {frames_with_utterances} frames with utterances")
                    print(f"Skipped {total_frames - frames_with_utterances} frames without utterances")
                
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                continue
    
    print(f"\nCreated frame-to-utterance mapping CSV file: {output_file}")

if __name__ == '__main__':
    process_xml_files() 