import pandas as pd
import numpy as np
import re

def analyze_dataset(df, filter_slight=False):
    """Analyze the dataset, optionally filtering out 'slightly' expressions"""
    total_sentences = len(df)

    def extract_timing(expression_str):
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
        eyebrows = extract_timing(str(row['face_eye_brows']) if pd.notna(row['face_eye_brows']) else '')
        mouth = extract_timing(str(row['face_mouth']) if pd.notna(row['face_mouth']) else '')
        cheeks = extract_timing(str(row['face_cheeks']) if pd.notna(row['face_cheeks']) else '')
        
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
            'num_cheeks': len(cheeks)
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
    avg_expressions = np.mean([r['total_expressions'] for r in results])
    avg_distinct = np.mean([r['total_distinct'] for r in results])
    avg_asl_gloss = np.mean(df['count_asl_gloss'])
    expression_to_gloss_ratio = avg_expressions / avg_asl_gloss if avg_asl_gloss > 0 else 0

    # Distribution by type
    total_eyebrows = sum(r['num_eyebrows'] > 0 for r in results)
    total_mouth = sum(r['num_mouth'] > 0 for r in results)
    total_cheeks = sum(r['num_cheeks'] > 0 for r in results)

    print(f"\n{'=' * 20} EXPRESSION ANALYSIS {'=' * 20}")
    print(f"Total sentences analyzed: {total_sentences}")
    print(f"Sentences with facial expressions: {total_with_expressions} ({(total_with_expressions/total_sentences)*100:.1f}%)")

    print("\nSequential expressions by area:")
    print(f"Eyebrows: {total_with_multiple_sequential['eyebrows']} sentences with >=2 sequential ({(total_with_multiple_sequential['eyebrows']/total_sentences)*100:.1f}%), avg {avg_sequential['eyebrows']:.1f} per sentence")
    print(f"Mouth: {total_with_multiple_sequential['mouth']} sentences with >=2 sequential ({(total_with_multiple_sequential['mouth']/total_sentences)*100:.1f}%), avg {avg_sequential['mouth']:.1f} per sentence")
    print(f"Cheeks: {total_with_multiple_sequential['cheeks']} sentences with >=2 sequential ({(total_with_multiple_sequential['cheeks']/total_sentences)*100:.1f}%), avg {avg_sequential['cheeks']:.1f} per sentence")
    print(f"Combined: {total_with_multiple_sequential['total']} sentences with >=2 sequential ({(total_with_multiple_sequential['total']/total_sentences)*100:.1f}%), avg {avg_sequential['total']:.1f} per sentence")

    print(f"\nAverage facial expressions per sentence: {avg_expressions:.1f}")
    print(f"Average distinct facial expressions per sentence: {avg_distinct:.1f}")
    print(f"Average ASL gloss count per sentence: {avg_asl_gloss:.1f}")
    print(f"Ratio of facial expressions to ASL gloss: {expression_to_gloss_ratio:.2f}")

    print("\nDistribution of expression types:")
    print(f"Sentences with eyebrow expressions: {total_eyebrows} ({(total_eyebrows/total_sentences)*100:.1f}%)")
    print(f"Sentences with mouth expressions: {total_mouth} ({(total_mouth/total_sentences)*100:.1f}%)")
    print(f"Sentences with cheek expressions: {total_cheeks} ({(total_cheeks/total_sentences)*100:.1f}%)")

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
df = pd.read_csv('parsing/xml_csvs/facial_expressions_by_sentence.csv')

# Run analysis with all expressions
analyze_dataset(df, filter_slight=False)