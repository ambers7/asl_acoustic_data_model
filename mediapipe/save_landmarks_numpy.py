import numpy as np
import cv2
import mediapipe as mp
import os
import pandas as pd

# === MediaPipe setup ===
mp_holistic = mp.solutions.holistic

# === Paths ===
input_video_path = "mediapipe/videos/1-Introduction-SD.mov"
output_dir = "mediapipe/numpy_data"
utterance_map_path = "parsing/xml_csvs/frame_utterance_map.csv"  # Updated path
os.makedirs(output_dir, exist_ok=True)

def load_utterance_data():
    """Load and preprocess utterance mapping data"""
    try:
        utterance_df = pd.read_csv(utterance_map_path)
        # Rename columns to match our expected format
        utterance_df = utterance_df.rename(columns={
            '#': 'file_number',
            'utterance_id': 'utterance_id',
            'manual_signs': 'manual_signs',  
            'non_manual_signs': 'non_manual_signs'
        })
        print(f"Loaded utterance data with {len(utterance_df)} entries")
        print("Available columns:", utterance_df.columns.tolist())
        return utterance_df
    except Exception as e:
        print(f"Warning: Could not load utterance data: {e}")
        print("Current working directory:", os.getcwd())
        print("Looking for file:", os.path.abspath(utterance_map_path))
        return None

def process_video():
    """Process video and save landmarks as numpy arrays"""
    # Load utterance data first
    utterance_df = load_utterance_data()
    
    # Open video to get frame count
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_video_path}")
        return

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize MediaPipe
    print("\nInitializing MediaPipe Holistic...")
    holistic = mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    # Create a structured array to store all frame data
    # Define the dtype for our structured array
    dtype = [
        ('frame', np.int32),
        ('file_number', 'U20'),    # Increased size to handle larger numbers
        ('utterance_id', 'U20'),        # Unicode string max length 20
        ('manual_signs', 'U100'),       # Unicode string max length 100
        ('non_manual_signs', 'U200'),   # Increased size for longer non-manual sign descriptions
        
        # Pose landmarks (x, y, z, visibility for each landmark)
        ('pose_nose', np.float32, (4,)),           # landmark 0
        ('pose_left_eye_inner', np.float32, (4,)), # landmark 1
        ('pose_left_eye', np.float32, (4,)),       # landmark 2
        ('pose_left_eye_outer', np.float32, (4,)), # landmark 3
        ('pose_right_eye_inner', np.float32, (4,)),# landmark 4
        ('pose_right_eye', np.float32, (4,)),      # landmark 5
        ('pose_right_eye_outer', np.float32, (4,)),# landmark 6
        ('pose_left_ear', np.float32, (4,)),       # landmark 7
        ('pose_right_ear', np.float32, (4,)),      # landmark 8
        ('pose_mouth_left', np.float32, (4,)),     # landmark 9
        ('pose_mouth_right', np.float32, (4,)),    # landmark 10
        ('pose_left_shoulder', np.float32, (4,)),  # landmark 11
        ('pose_right_shoulder', np.float32, (4,)), # landmark 12
        ('pose_left_elbow', np.float32, (4,)),     # landmark 13
        ('pose_right_elbow', np.float32, (4,)),    # landmark 14
        ('pose_left_wrist', np.float32, (4,)),     # landmark 15
        ('pose_right_wrist', np.float32, (4,)),    # landmark 16
        ('pose_left_pinky', np.float32, (4,)),     # landmark 17
        ('pose_right_pinky', np.float32, (4,)),    # landmark 18
        ('pose_left_index', np.float32, (4,)),     # landmark 19
        ('pose_right_index', np.float32, (4,)),    # landmark 20
        ('pose_left_thumb', np.float32, (4,)),     # landmark 21
        ('pose_right_thumb', np.float32, (4,)),    # landmark 22
        ('pose_left_hip', np.float32, (4,)),       # landmark 23
        ('pose_right_hip', np.float32, (4,)),      # landmark 24
        ('pose_left_knee', np.float32, (4,)),      # landmark 25
        ('pose_right_knee', np.float32, (4,)),     # landmark 26
        ('pose_left_ankle', np.float32, (4,)),     # landmark 27
        ('pose_right_ankle', np.float32, (4,)),    # landmark 28
        ('pose_left_heel', np.float32, (4,)),      # landmark 29
        ('pose_right_heel', np.float32, (4,)),     # landmark 30
        ('pose_left_foot_index', np.float32, (4,)),# landmark 31
        ('pose_right_foot_index', np.float32, (4,)),# landmark 32

        # Face landmarks by region (each point has x, y, z, visibility)
        # Face oval (silhouette)
        *[('face_oval_' + str(i), np.float32, (4,)) for i in range(17)],
        
        # Eyebrows
        *[('left_eyebrow_' + str(i), np.float32, (4,)) for i in range(5)],  # 17-21
        *[('right_eyebrow_' + str(i), np.float32, (4,)) for i in range(5)], # 22-26
        *[('left_eyebrow_upper_' + str(i), np.float32, (4,)) for i in range(5)],  # 70-74
        *[('right_eyebrow_upper_' + str(i), np.float32, (4,)) for i in range(5)], # 75-79
        *[('left_eyebrow_lower_' + str(i), np.float32, (4,)) for i in range(5)],  # 80-84
        *[('right_eyebrow_lower_' + str(i), np.float32, (4,)) for i in range(5)], # 85-89
        
        # Eyes detailed
        *[('nose_bridge_' + str(i), np.float32, (4,)) for i in range(4)],   # 27-30
        *[('nose_tip_' + str(i), np.float32, (4,)) for i in range(5)],      # 31-35
        *[('left_eye_outline_' + str(i), np.float32, (4,)) for i in range(6)],  # 36-41
        *[('right_eye_outline_' + str(i), np.float32, (4,)) for i in range(6)], # 42-47
        *[('outer_lips_' + str(i), np.float32, (4,)) for i in range(12)],   # 48-59
        *[('inner_lips_' + str(i), np.float32, (4,)) for i in range(8)],    # 60-67
        *[('pupil_center_' + str(i), np.float32, (4,)) for i in range(2)],  # 68-69
        
        # Eyes additional detail
        *[('left_eye_upper0_' + str(i), np.float32, (4,)) for i in range(6)],   # 90-95
        *[('right_eye_upper0_' + str(i), np.float32, (4,)) for i in range(6)],  # 96-101
        *[('left_eye_lower0_' + str(i), np.float32, (4,)) for i in range(6)],   # 102-107
        *[('right_eye_lower0_' + str(i), np.float32, (4,)) for i in range(6)],  # 108-113
        *[('left_eye_upper1_' + str(i), np.float32, (4,)) for i in range(6)],   # 114-119
        *[('right_eye_upper1_' + str(i), np.float32, (4,)) for i in range(6)],  # 120-125
        *[('left_eye_lower1_' + str(i), np.float32, (4,)) for i in range(6)],   # 126-131
        *[('right_eye_lower1_' + str(i), np.float32, (4,)) for i in range(6)],  # 132-137
        *[('left_eye_upper2_' + str(i), np.float32, (4,)) for i in range(6)],   # 138-143
        *[('right_eye_upper2_' + str(i), np.float32, (4,)) for i in range(6)],  # 144-149
        *[('left_eye_lower2_' + str(i), np.float32, (4,)) for i in range(6)],   # 150-155
        *[('right_eye_lower2_' + str(i), np.float32, (4,)) for i in range(6)],  # 156-161
        
        # Nose detailed
        *[('nose_bridge_detailed_' + str(i), np.float32, (4,)) for i in range(6)],   # 162-167
        *[('nose_tip_detailed_' + str(i), np.float32, (4,)) for i in range(6)],      # 168-173
        *[('nose_bottom_' + str(i), np.float32, (4,)) for i in range(6)],            # 174-179
        *[('nose_right_outline_' + str(i), np.float32, (4,)) for i in range(6)],     # 180-185
        *[('nose_left_outline_' + str(i), np.float32, (4,)) for i in range(6)],      # 186-191
        
        # Lips detailed
        *[('upper_lip_top_' + str(i), np.float32, (4,)) for i in range(12)],     # 192-203
        *[('upper_lip_bottom_' + str(i), np.float32, (4,)) for i in range(12)],  # 204-215
        *[('lower_lip_top_' + str(i), np.float32, (4,)) for i in range(12)],     # 216-227
        *[('lower_lip_bottom_' + str(i), np.float32, (4,)) for i in range(12)],  # 228-239
        
        # Mouth cavity
        *[('mouth_cavity_upper_' + str(i), np.float32, (4,)) for i in range(12)], # 240-251
        *[('mouth_cavity_lower_' + str(i), np.float32, (4,)) for i in range(12)], # 252-263
        
        # Face cheeks and additional contours
        *[('left_cheek_upper_' + str(i), np.float32, (4,)) for i in range(6)],   # 264-269
        *[('right_cheek_upper_' + str(i), np.float32, (4,)) for i in range(6)],  # 270-275
        *[('left_cheek_lower_' + str(i), np.float32, (4,)) for i in range(6)],   # 276-281
        *[('right_cheek_lower_' + str(i), np.float32, (4,)) for i in range(6)],  # 282-287
        
        # Face additional contours
        *[('face_contour_upper_' + str(i), np.float32, (4,)) for i in range(12)],   # 288-299
        *[('face_contour_lower_' + str(i), np.float32, (4,)) for i in range(12)],   # 300-311
        *[('face_contour_cheeks_' + str(i), np.float32, (4,)) for i in range(12)],  # 312-323
        
        # Detailed eye regions
        *[('left_eye_creases_' + str(i), np.float32, (4,)) for i in range(12)],    # 324-335
        *[('right_eye_creases_' + str(i), np.float32, (4,)) for i in range(12)],   # 336-347
        *[('left_eye_wrinkles_' + str(i), np.float32, (4,)) for i in range(12)],   # 348-359
        *[('right_eye_wrinkles_' + str(i), np.float32, (4,)) for i in range(12)],  # 360-371
        
        # Additional facial features
        *[('forehead_upper_' + str(i), np.float32, (4,)) for i in range(12)],      # 372-383
        *[('forehead_lower_' + str(i), np.float32, (4,)) for i in range(12)],      # 384-395
        *[('temple_left_' + str(i), np.float32, (4,)) for i in range(12)],         # 396-407
        *[('temple_right_' + str(i), np.float32, (4,)) for i in range(12)],        # 408-419
        
        # Detailed nose features
        *[('nose_bridge_wrinkles_' + str(i), np.float32, (4,)) for i in range(12)],    # 420-431
        *[('nose_side_wrinkles_' + str(i), np.float32, (4,)) for i in range(12)],      # 432-443
        *[('nose_tip_detailed_wrinkles_' + str(i), np.float32, (4,)) for i in range(12)], # 444-455
        
        # Additional mouth features
        *[('mouth_corners_detailed_' + str(i), np.float32, (4,)) for i in range(12)],   # 456-467
        
        # Iris landmarks
        ('left_iris', np.float32, (4,)),    # 468
        ('right_iris', np.float32, (4,)),   # 469
        
        # Additional face mesh points
        *[('left_eye_mesh_extra_' + str(i), np.float32, (4,)) for i in range(4)],   # 470-473
        *[('right_eye_mesh_extra_' + str(i), np.float32, (4,)) for i in range(4)],  # 474-477
        
        # Left hand landmarks (21 points, each with x,y,z,visibility)
        ('left_hand', np.float32, (21, 4)),
        # Right hand landmarks (21 points, each with x,y,z,visibility)
        ('right_hand', np.float32, (21, 4))
    ]

    # Initialize the structured array
    frame_data = np.zeros(total_frames, dtype=dtype)
    # Fill in frame numbers
    frame_data['frame'] = np.arange(total_frames)
    
    # Fill in utterance data if available
    if utterance_df is not None:
        print("\nProcessing utterance data...")
        debug_count = 0  # Counter for debug printing
        for _, row in utterance_df.iterrows():
            frame = row['frame']
            if frame < total_frames:
                # Convert all values to strings to avoid type mismatches
                frame_data[frame]['file_number'] = str(row.get('#', ''))
                frame_data[frame]['utterance_id'] = str(row.get('utterance_id', ''))
                frame_data[frame]['manual_signs'] = str(row.get('manual_signs', ''))
                frame_data[frame]['non_manual_signs'] = str(row.get('non_manual_signs', ''))
                
                # Debug print for the first few frames with data
                if debug_count < 5:  # Print first 5 entries
                    print(f"\nFrame {frame} utterance data:")
                    print(f"Number: {frame_data[frame]['file_number']}")
                    print(f"ID: {frame_data[frame]['utterance_id']}")
                    print(f"Manual: {frame_data[frame]['manual_signs']}")
                    print(f"Non-manual: {frame_data[frame]['non_manual_signs']}")
                    debug_count += 1
    
    # Process frames for landmarks
    print("\nProcessing video frames for landmarks...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = holistic.process(rgb)
        
        # Store pose landmarks
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # Get the field name for this pose landmark
                pose_fields = [name for name in frame_data.dtype.names if name.startswith('pose_')]
                if idx < len(pose_fields):
                    frame_data[frame_idx][pose_fields[idx]] = [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ]
        
        # Store face landmarks
        if results.face_landmarks:
            for idx, landmark in enumerate(results.face_landmarks.landmark):
                # Find the corresponding field name for this face landmark
                face_fields = [name for name in frame_data.dtype.names 
                             if any(name.startswith(prefix) for prefix in [
                                 'face_', 'left_eye', 'right_eye', 'nose_', 'mouth_',
                                 'upper_lip', 'lower_lip', 'left_cheek', 'right_cheek',
                                 'temple_', 'forehead_'
                             ])]
                if idx < len(face_fields):
                    frame_data[frame_idx][face_fields[idx]] = [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ]
        
        # Store left hand landmarks
        if results.left_hand_landmarks:
            left_hand = np.zeros((21, 4))
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                left_hand[idx] = [landmark.x, landmark.y, landmark.z, 1.0]
            frame_data[frame_idx]['left_hand'] = left_hand
        
        # Store right hand landmarks
        if results.right_hand_landmarks:
            right_hand = np.zeros((21, 4))
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                right_hand[idx] = [landmark.x, landmark.y, landmark.z, 1.0]
            frame_data[frame_idx]['right_hand'] = right_hand
        
        # Show progress
        if frame_idx % 10 == 0:
            print(f"\rProcessing frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)", end="")
        
        frame_idx += 1
    
    print("\nLandmark processing complete!")
    
    # Save the data
    print("\nSaving data...")
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_video_path))[0]}_landmarks.npz")
    np.savez_compressed(output_path, frame_data=frame_data)
    
    print(f"\nSaved all data to: {output_path}")
    print("\nData structure:")
    print("Number of frames:", len(frame_data))
    print("Fields per frame:", frame_data.dtype.names)
    
    # Example of how to access the data
    print("\nExample data from first frame with detections:")
    example_frame = frame_data[0]
    print(f"Frame number: {example_frame['frame']}")
    print(f"Utterance ID: {example_frame['utterance_id']}")
    print(f"Manual signs: {example_frame['manual_signs']}")
    if np.any(example_frame['left_hand']):
        print("Left hand detected")
    if np.any(example_frame['right_hand']):
        print("Right hand detected")
    
    # Cleanup
    cap.release()
    holistic.close()

if __name__ == '__main__':
    process_video() 