import xml.etree.ElementTree as ET
import glob
import csv
from collections import Counter

# Get all XML files in the xml_files directory
xml_files = glob.glob('xml_files/*.xml')

# Counter for hash words
hash_word_counter = Counter()

# Process each XML file
for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Find all SIGN elements
        for sign in root.findall('.//SIGN'):
            label_elem = sign.find('LABEL')
            if label_elem is not None and label_elem.text:
                # Clean the label text
                label = label_elem.text.strip("'")
                # Check if it starts with #
                if label.startswith('#'):
                    hash_word_counter[label] += 1
                
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")
        continue

# Write results to CSV
output_file = 'xml_csvs/hash_gloss_counts.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['gloss_word', 'count'])
    
    # Sort by count in descending order
    for word, count in sorted(hash_word_counter.items(), key=lambda x: x[1], reverse=True):
        writer.writerow([word, count])

print(f"\nCreated hash gloss word count CSV file: {output_file}")
print(f"Total unique hash gloss words found: {len(hash_word_counter)}")
print(f"Total hash gloss word occurrences: {sum(hash_word_counter.values())}")

# Print the top 10 most frequent hash words
print("\nTop 10 most frequent hash gloss words:")
for rank, (word, count) in enumerate(sorted(hash_word_counter.items(), key=lambda x: x[1], reverse=True)[:10], 1):
    print(f"  {rank}. {word}: {count}") 