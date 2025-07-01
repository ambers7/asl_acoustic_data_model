import xml.etree.ElementTree as ET
import csv
import os
import glob

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

# Get all XML files in the xml_files directory
xml_files = glob.glob('xml_files/*.xml')

if not xml_files:
    print("No XML files found in xml_files/ directory.")
    exit()

csv_file = 'xml_csvs/asl_public_dataset.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ['#', 'utterance_id', 'translation', 'asl_gloss'] + list(feature_map.keys())
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
                if translation_elem is not None and translation_elem.text is not None:
                    translation = translation_elem.text.strip("'")

                labels = []
                manuals = utterance.find('MANUALS')
                if manuals is not None:
                    for sign in manuals.findall('SIGN'):
                        label_elem = sign.find('LABEL')
                        if label_elem is not None and label_elem.text:
                            labels.append(label_elem.text.strip("'"))

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

                writer.writerow([collection_id, utterance_id, translation, ';'.join(labels)] + feature_values)

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue

print(f"CSV file '{csv_file}' created.") 