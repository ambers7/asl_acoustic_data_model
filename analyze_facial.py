import pandas as pd
import numpy as np
import re

def frames_overlap(start1, end1, start2, end2):
    """Check if two frame ranges overlap."""
    return not (end1 < start2 or start1 > end2)

def extract_timing(expression_str, filter_slight=False):
    """Extract timing information from expression string.
    Example input: "raised(10-20);lowered(25-35)"
    Returns list of tuples: [(expression, start, end), ...]"""
    if not isinstance(expression_str, str) or not expression_str:
        return []
    
    expressions = []
    for expr in expression_str.split(';'):
        if not expr:
            continue
        match = re.match(r'(.*?)\((\d+)-(\d+)\)', expr)
        if match:
            expression, start, end = match.groups()
            expression = expression.strip()
            
            # Filter out "slightly" expressions if requested
            if filter_slight and ('slightly raised' in expression or 'slightly lowered' in expression):
                continue
                
            expressions.append((expression, int(start), int(end)))
    return expressions

def analyze_dataset(df, filter_slight=False):
    """Analyze the dataset, optionally filtering out 'slightly' expressions"""
    total_sentences = len(df)

    def get_distinct_expressions(expression_str):
        """Get unique expressions ignoring timing."""
        if not isinstance(expression_str, str) or not expression_str:
            return set()
        
        expressions = set()
        for expr in expression_str.split(';'):
            if not expr:
                continue
            match = re.match(r'(.*?)\((\d+)-(\d+)\)', expr)
            if match:
                expression = match.group(1).strip()
                expressions.add(expression)
        return expressions

    def count_sequential_expressions(row):
        """Count sequential expressions for each area and combined"""
        # Get expressions for each area
        eyebrows = extract_timing(str(row['face_eye_brows']) if pd.notna(row['face_eye_brows']) else '')
        mouth = extract_timing(str(row['face_mouth']) if pd.notna(row['face_mouth']) else '')
        cheeks = extract_timing(str(row['face_cheeks']) if pd.notna(row['face_cheeks']) else '')
        
        # Count expressions in each area
        sequential_eyebrows = len(eyebrows)  # All expressions in same area are sequential
        sequential_mouth = len(mouth)
        sequential_cheeks = len(cheeks)
        
        # For combined count, need to account for non-sequential expressions between areas
        frame_areas = {}  # frame -> set of areas active
        for expr, start, end in eyebrows:
            frame = (start, end)
            if frame not in frame_areas:
                frame_areas[frame] = set()
            frame_areas[frame].add('eyebrows')
            
        for expr, start, end in mouth:
            frame = (start, end)
            if frame not in frame_areas:
                frame_areas[frame] = set()
            frame_areas[frame].add('mouth')
            
        for expr, start, end in cheeks:
            frame = (start, end)
            if frame not in frame_areas:
                frame_areas[frame] = set()
            frame_areas[frame].add('cheeks')
        
        # Count non-sequential expressions (different areas moving at same time)
        non_sequential = 0
        for frame, areas in frame_areas.items():
            if len(areas) > 1:  # If multiple areas active at same time
                non_sequential += len(areas) - 1  # Count all but one as non-sequential
                
        total_sequential = sequential_eyebrows + sequential_mouth + sequential_cheeks - non_sequential
        
        return {
            'eyebrows': sequential_eyebrows,
            'mouth': sequential_mouth,
            'cheeks': sequential_cheeks,
            'total': total_sequential
        }

    def analyze_facial_expressions(row):
        # Get expressions for each feature with timing
        eyebrows = extract_timing(str(row['face_eye_brows']) if pd.notna(row['face_eye_brows']) else '', filter_slight)
        mouth = extract_timing(str(row['face_mouth']) if pd.notna(row['face_mouth']) else '', filter_slight)
        cheeks = extract_timing(str(row['face_cheeks']) if pd.notna(row['face_cheeks']) else '', filter_slight)
        
        # Get distinct expressions
        distinct_eyebrows = get_distinct_expressions(str(row['face_eye_brows']) if pd.notna(row['face_eye_brows']) else '')
        distinct_mouth = get_distinct_expressions(str(row['face_mouth']) if pd.notna(row['face_mouth']) else '')
        distinct_cheeks = get_distinct_expressions(str(row['face_cheeks']) if pd.notna(row['face_cheeks']) else '')
        
        # Count sequential expressions
        sequential_counts = count_sequential_expressions(row)
        
        # Total facial expressions
        total_expressions = len(eyebrows) + len(mouth) + len(cheeks)
        
        # Total distinct expressions
        total_distinct = len(distinct_eyebrows) + len(distinct_mouth) + len(distinct_cheeks)
        
        # Check for valid frames (not placeholders where start=end)
        valid_expressions = []
        for area_exprs in [eyebrows, mouth, cheeks]:
            for expr, start, end in area_exprs:
                if start != end:  # Only count if not a placeholder
                    valid_expressions.append((expr, start, end))
        
        # Check for overlaps among valid expressions
        has_overlap = False
        if len(valid_expressions) >= 2:
            for i in range(len(valid_expressions)):
                for j in range(i + 1, len(valid_expressions)):
                    _, start1, end1 = valid_expressions[i]
                    _, start2, end2 = valid_expressions[j]
                    if frames_overlap(start1, end1, start2, end2):
                        has_overlap = True
                        break
                if has_overlap:
                    break
        
        return {
            'has_expressions': bool(eyebrows or mouth or cheeks),
            'sequential_eyebrows': sequential_counts['eyebrows'],
            'sequential_mouth': sequential_counts['mouth'],
            'sequential_cheeks': sequential_counts['cheeks'],
            'sequential_total': sequential_counts['total'],
            'total_expressions': total_expressions,
            'total_distinct': total_distinct,
            'num_eyebrows': len(eyebrows),
            'num_mouth': len(mouth),
            'num_cheeks': len(cheeks),
            'has_valid_frames': bool(valid_expressions),
            'has_overlap': has_overlap
        }

    # Analyze each sentence
    results = [analyze_facial_expressions(row) for _, row in df.iterrows()]

    # Calculate statistics
    total_with_expressions = sum(r['has_expressions'] for r in results)
    total_with_multiple_sequential = {
        'eyebrows': sum(r['sequential_eyebrows'] >= 2 for r in results),
        'mouth': sum(r['sequential_mouth'] >= 2 for r in results),
        'cheeks': sum(r['sequential_cheeks'] >= 2 for r in results),
        'total': sum(r['sequential_total'] >= 2 for r in results)
    }
    avg_sequential = {
        'eyebrows': np.mean([r['sequential_eyebrows'] for r in results]),
        'mouth': np.mean([r['sequential_mouth'] for r in results]),
        'cheeks': np.mean([r['sequential_cheeks'] for r in results]),
        'total': np.mean([r['sequential_total'] for r in results])
    }
    # Calculate means and standard deviations
    distinct_expressions = [r['total_distinct'] for r in results]
    asl_gloss_counts = df['count_asl_gloss'].values
    
    avg_expressions = np.mean([r['total_expressions'] for r in results])
    avg_distinct = np.mean(distinct_expressions)
    std_distinct = np.std(distinct_expressions)
    avg_asl_gloss = np.mean(asl_gloss_counts)
    std_asl_gloss = np.std(asl_gloss_counts)
    expression_to_gloss_ratio = avg_expressions / avg_asl_gloss if avg_asl_gloss > 0 else 0

    # Distribution by type
    total_eyebrows = sum(r['num_eyebrows'] > 0 for r in results)
    total_mouth = sum(r['num_mouth'] > 0 for r in results)
    total_cheeks = sum(r['num_cheeks'] > 0 for r in results)

    print(f"\n{'=' * 20} EXPRESSION ANALYSIS {'=' * 20}")
    print(f"Total sentences analyzed: {total_sentences}")
    print(f"Sentences with facial expressions: {total_with_expressions} ({(total_with_expressions/total_sentences)*100:.1f}%)")

    print("\nSequential expressions by area:")
    print(f"Eyebrows: {total_with_multiple_sequential['eyebrows']} sentences with >=2 sequential ({(total_with_multiple_sequential['eyebrows']/total_sentences)*100}%), avg {avg_sequential['eyebrows']} per sentence")
    print(f"Mouth: {total_with_multiple_sequential['mouth']} sentences with >=2 sequential ({(total_with_multiple_sequential['mouth']/total_sentences)*100}%), avg {avg_sequential['mouth']} per sentence")
    print(f"Cheeks: {total_with_multiple_sequential['cheeks']} sentences with >=2 sequential ({(total_with_multiple_sequential['cheeks']/total_sentences)*100}%), avg {avg_sequential['cheeks']} per sentence")
    print(f"Combined: {total_with_multiple_sequential['total']} sentences with >=2 sequential ({(total_with_multiple_sequential['total']/total_sentences)*100}%), avg {avg_sequential['total']} per sentence")

    print(f"\nAverage facial expressions per sentence: {avg_expressions}")
    print(f"Average distinct facial expressions per sentence: {avg_distinct} (std: {std_distinct})")
    print(f"Average ASL gloss count per sentence: {avg_asl_gloss} (std: {std_asl_gloss})")
    print(f"Ratio of facial expressions to ASL gloss: {expression_to_gloss_ratio}")

    # Count sentences with valid frames and overlaps
    total_with_valid_frames = sum(r['has_valid_frames'] for r in results)
    total_with_overlaps = sum(r['has_overlap'] for r in results)
    
    print("\nFrame Analysis:")
    print(f"Total sentences with valid frame labels (not placeholders): {total_with_valid_frames} ({(total_with_valid_frames/total_sentences)*100}%)")
    print(f"Of those, sentences with at least one overlap: {total_with_overlaps} ({(total_with_overlaps/total_with_valid_frames)*100 if total_with_valid_frames > 0 else 0}%)")

    print("\nDistribution of expression types:")
    print(f"Sentences with eyebrow expressions: {total_eyebrows} ({(total_eyebrows/total_sentences)*100}%)")
    print(f"Sentences with mouth expressions: {total_mouth} ({(total_mouth/total_sentences)*100}%)")
    print(f"Sentences with cheek expressions: {total_cheeks} ({(total_cheeks/total_sentences)*100}%)")

    # Example of a sentence with multiple sequential expressions
    print("\nExample of a sentence with multiple sequential expressions:")
    for idx, row in df.iterrows():
        sequential_counts = count_sequential_expressions(row)
        if sequential_counts['total'] >= 2:
            print(f"\nTranslation: {row['translation']}")
            print(f"ASL Gloss: {row['asl_gloss']}")
            print(f"ASL Gloss Count: {row['count_asl_gloss']}")
            print(f"Eyebrows: {row['face_eye_brows'] if pd.notna(row['face_eye_brows']) else 'none'}")
            print(f"Mouth: {row['face_mouth'] if pd.notna(row['face_mouth']) else 'none'}")
            print(f"Cheeks: {row['face_cheeks'] if pd.notna(row['face_cheeks']) else 'none'}")
            print(f"Sequential expressions by area:")
            print(f"  Eyebrows: {sequential_counts['eyebrows']}")
            print(f"  Mouth: {sequential_counts['mouth']}")
            print(f"  Cheeks: {sequential_counts['cheeks']}")
            print(f"  Combined total: {sequential_counts['total']}")
            break

# Read the CSV file
df = pd.read_csv('parsing/xml_csvs/facial_expressions.csv')

# Function to count sentences in a text
def count_sentences(text):
    if not isinstance(text, str):
        return 0
    # Split on period, exclamation mark, or question mark
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings and count non-empty sentences
    return len([s for s in sentences if s.strip()])

# Analyze sentences per utterance
df['sentence_count'] = df['translation'].apply(count_sentences)
utterance_sentences = df.groupby('utterance_id')['sentence_count'].first()  # Take first row for each utterance since they're identical

min_sentences = utterance_sentences.min()
max_sentences = utterance_sentences.max()
mean_sentences = utterance_sentences.mean()
std_sentences = utterance_sentences.std()

print("\n=== Sentences per Utterance Analysis ===")
print(f"Minimum sentences in an utterance: {min_sentences}")
print(f"Maximum sentences in an utterance: {max_sentences}")
print(f"Average sentences per utterance: {mean_sentences}")
print(f"Standard deviation: {std_sentences}")
print("\nDistribution of sentences per utterance:")
print(utterance_sentences.value_counts().sort_index())

# Function to get overlapping expressions from a row
def get_overlapping_expressions(row):
    expressions = []
    # Get expressions with timing from each area
    for field in ['face_eye_brows', 'face_mouth', 'face_cheeks']:
        if pd.notna(row[field]):
            area_exprs = extract_timing(str(row[field]), filter_slight=False)
            for expr, start, end in area_exprs:
                if start != end:  # Only include non-placeholder frames
                    expressions.append((field, expr, start, end))
    
    # Find overlapping pairs
    overlaps = []
    for i in range(len(expressions)):
        for j in range(i + 1, len(expressions)):
            area1, expr1, start1, end1 = expressions[i]
            area2, expr2, start2, end2 = expressions[j]
            if frames_overlap(start1, end1, start2, end2):
                overlaps.append((area1, expr1, start1, end1, area2, expr2, start2, end2))
    return overlaps

# Print examples of overlapping expressions
print("\nExamples of utterances with overlapping facial expressions:")
examples_shown = 0
for idx, row in df.iterrows():
    overlaps = get_overlapping_expressions(row)
    if overlaps:
        print(f"\n=== Example {examples_shown + 1} ===")
        print(f"Utterance ID: {row['utterance_id']}")
        print(f"Translation: {row['translation']}")
        print(f"ASL Gloss: {row['asl_gloss']}")
        
        print("\nAll facial expressions in this utterance:")
        print(f"Eyebrows: {row['face_eye_brows'] if pd.notna(row['face_eye_brows']) else 'none'}")
        print(f"Mouth: {row['face_mouth'] if pd.notna(row['face_mouth']) else 'none'}")
        print(f"Cheeks: {row['face_cheeks'] if pd.notna(row['face_cheeks']) else 'none'}")
        
        print("\nOverlapping expressions:")
        for area1, expr1, start1, end1, area2, expr2, start2, end2 in overlaps:
            print(f"- {area1.split('face_')[1]}: {expr1} ({start1}-{end1})")
            print(f"  overlaps with")
            print(f"  {area2.split('face_')[1]}: {expr2} ({start2}-{end2})")
        
        examples_shown += 1
        if examples_shown >= 3:  # Show 3 examples
            break

# Run analysis with all expressions
analyze_dataset(df, filter_slight=False)