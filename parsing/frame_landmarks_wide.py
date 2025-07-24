import pandas as pd
import os
import numpy as np

def process_landmark_file(landmark_file, utterance_df, output_file, first_file=False):
    """Process a single landmark file and append to output CSV"""
    print(f"Reading {landmark_file}...")
    df = pd.read_csv(landmark_file)
    
    # Get frames in this file
    frames = sorted(df['frame'].unique())
    wide_df = pd.DataFrame({'frame': frames})
    
    # For each unique landmark type (body part)
    unique_landmarks = sorted(df['type'].unique())
    for landmark in unique_landmarks:
        # Filter data for this landmark
        landmark_data = df[df['type'] == landmark]
        
        # Create norm_coords array [x_norm, y_norm, z_norm]
        norm_coords = landmark_data.groupby('frame').apply(
            lambda x: x[['x_norm', 'y_norm', 'z_norm']].values[0].tolist()
            if len(x) > 0 else None
        )
        wide_df[f"{landmark}_norm_coords"] = wide_df['frame'].map(norm_coords)
        
        # Create pixel_coords array [x_px, y_px]
        pixel_coords = landmark_data.groupby('frame').apply(
            lambda x: x[['x_px', 'y_px']].values[0].tolist()
            if len(x) > 0 else None
        )
        wide_df[f"{landmark}_pixel_coords"] = wide_df['frame'].map(pixel_coords)
    
    # Merge with utterance data
    final_df = pd.merge(
        wide_df,
        utterance_df,
        on='frame',
        how='left'
    )
    
    # Reorder columns to put utterance data first
    utterance_cols = ['#', 'utterance_id', 'manual_signs', 'non_manual_signs', 'frame']
    landmark_cols = [col for col in final_df.columns if col not in utterance_cols]
    final_df = final_df[utterance_cols + landmark_cols]
    
    # Write to CSV (append mode if not first file)
    final_df.to_csv(output_file, index=False, mode='w' if first_file else 'a', header=first_file)
    print(f"Processed {len(frames)} frames")
    print(f"Total columns: {len(final_df.columns)}")
    if first_file:
        print("\nColumn order:")
        print(f"First columns: {', '.join(utterance_cols)}")
        print(f"Followed by landmark columns like: {', '.join(landmark_cols[:3])}...")
    
    # Free memory
    del df, wide_df, final_df
    import gc
    gc.collect()

def transform_landmarks_to_wide():
    """
    Transform landmarks CSV and combine with frame_utterance_map.csv where:
    - Each row is a frame
    - Each body part has two coordinate columns:
        - norm_coords: [x_norm, y_norm, z_norm]
        - pixel_coords: [x_px, y_px]
    Processes and writes one part file at a time to save memory.
    """
    # Get script directory and construct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, '..', 'mediapipe', 'mediapipe_csvs')
    output_dir = os.path.join(script_dir, 'landmarks_wide')
    utterance_map_path = os.path.join(script_dir, 'xml_csvs/frames', 'frame_utterance_map.csv')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read utterance mapping
    utterance_df = pd.read_csv(utterance_map_path)
    print(f"\nRead utterance mapping with {len(utterance_df)} rows")
    
    # Process the landmarks directory
    video_dir = "1-Introduction-SD"  # The folder containing landmark CSVs
    video_path = os.path.join(input_dir, video_dir)
    if not os.path.isdir(video_path):
        print(f"Error: Directory not found: {video_path}")
        return
        
    print(f"\nProcessing video directory: {video_dir}")
    # Use the landmarks folder name for output file
    output_file = os.path.join(output_dir, f"frame_{video_dir}.csv")
    
    # Process each part file
    first_file = True
    for file in sorted(os.listdir(video_path)):
        if file.startswith('landmarks_part') and file.endswith('.csv'):
            file_path = os.path.join(video_path, file)
            process_landmark_file(file_path, utterance_df, output_file, first_file)
            first_file = False
    
    print(f"\nCompleted processing all parts to: {output_file}")

if __name__ == '__main__':
    transform_landmarks_to_wide() 