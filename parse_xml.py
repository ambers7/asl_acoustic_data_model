import xml.etree.ElementTree as ET
import csv

xml_file = 'sample.xml'
tree = ET.parse(xml_file)
root = tree.getroot()

csv_file = 'xml_csvs/utterances_manual_labels.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'translation', 'asl_gloss'])

    for utterance in root.findall('.//UTTERANCE'):
        utterance_id = utterance.get('ID', '').strip("'")
        translation_elem = utterance.find('TRANSLATION')
        translation = translation_elem.text.strip("'") if translation_elem is not None else ''

        labels = []
        manuals = utterance.find('MANUALS')
        if manuals is not None:
            for sign in manuals.findall('SIGN'):
                label_elem = sign.find('LABEL')
                if label_elem is not None and label_elem.text:
                    labels.append(label_elem.text.strip("'"))
        writer.writerow([utterance_id, translation, ';'.join(labels)])

print(f"CSV file '{csv_file}' created.") 