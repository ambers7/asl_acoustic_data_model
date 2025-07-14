import cv2
import numpy as np
import mediapipe as mp
import os
import torch
import tensorflow as tf

def setup_gpu(gpu_id=0):
    """Setup GPU configuration for PyTorch and TensorFlow"""
    if torch.cuda.is_available():
        # Set PyTorch device
        torch.cuda.set_device(gpu_id)
        # Enable TensorFlow GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        print(f"Using GPU {gpu_id} for PyTorch/TensorFlow operations")
        return True
    print("GPU not available, using CPU")
    return False

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

def draw_landmarks_on_frame(frame, face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks):
    """Draw landmarks on a single frame"""
    # Draw face mesh
    if face_landmarks is not None:
        # Get the number of landmarks we actually have
        num_landmarks = len(face_landmarks.landmark)
        
        # Filter FACEMESH_TESSELATION to only include valid connections
        valid_tesselation = [
            connection for connection in mp_face_mesh.FACEMESH_TESSELATION
            if connection[0] < num_landmarks and connection[1] < num_landmarks
        ]
        
        # Filter FACEMESH_CONTOURS to only include valid connections
        valid_contours = [
            connection for connection in mp_face_mesh.FACEMESH_CONTOURS
            if connection[0] < num_landmarks and connection[1] < num_landmarks
        ]
        
        # Draw tesselation
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=valid_tesselation,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        # Draw contours
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=valid_contours,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )

    # Draw pose landmarks
    if pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Draw hand landmarks
    if left_hand_landmarks is not None:
        mp_drawing.draw_landmarks(
            frame,
            left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )
    if right_hand_landmarks is not None:
        mp_drawing.draw_landmarks(
            frame,
            right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )

def create_landmark_list():
    """Create an empty landmark list using the current MediaPipe API"""
    class Landmark:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.visibility = 0.0
            self._has_visibility = False
        
        def HasField(self, field_name):
            if field_name == 'visibility':
                return self._has_visibility
            return False

        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == 'visibility' and value != 0.0:
                super().__setattr__('_has_visibility', True)

    class NormalizedLandmarkList:
        def __init__(self):
            self.landmark = []
        
        def add(self):
            landmark = Landmark()
            self.landmark.append(landmark)
            return landmark

    return NormalizedLandmarkList()

def create_landmark_list_from_points(points):
    """Create a landmark list from numpy array points"""
    if points is None or np.all(points == 0):
        return None
    landmark_list = create_landmark_list()
    for point in points:
        if not np.all(point == 0):
            landmark = landmark_list.add()
            landmark.x = float(point[0])
            landmark.y = float(point[1])
            landmark.z = float(point[2])
            landmark.visibility = float(point[3])
    return landmark_list if len(landmark_list.landmark) > 0 else None

def add_landmark_point(landmark_list, point_data):
    """Add a single landmark point to the list"""
    if point_data is not None and not np.all(point_data == 0):
        landmark = landmark_list.add()
        landmark.x = float(point_data[0])
        landmark.y = float(point_data[1])
        landmark.z = float(point_data[2])
        landmark.visibility = float(point_data[3])

def create_face_landmarks_from_frame(frame_landmarks):
    """Create face landmarks in the correct order matching MediaPipe's expected indices"""
    if frame_landmarks is None:
        return None
        
    face_landmarks = create_landmark_list()
    
    # Add landmarks in the exact order they were saved
    # First add face oval (0-16)
    for i in range(17):
        field_name = f'face_oval_{i}'
        if field_name in frame_landmarks.dtype.names:
            add_landmark_point(face_landmarks, frame_landmarks[field_name])
    
    # Add eyebrows (17-26)
    for i in range(5):
        field_name = f'left_eyebrow_{i}'
        if field_name in frame_landmarks.dtype.names:
            add_landmark_point(face_landmarks, frame_landmarks[field_name])
    for i in range(5):
        field_name = f'right_eyebrow_{i}'
        if field_name in frame_landmarks.dtype.names:
            add_landmark_point(face_landmarks, frame_landmarks[field_name])
    
    # Add nose (27-35)
    for i in range(4):
        field_name = f'nose_bridge_{i}'
        if field_name in frame_landmarks.dtype.names:
            add_landmark_point(face_landmarks, frame_landmarks[field_name])
    for i in range(5):
        field_name = f'nose_tip_{i}'
        if field_name in frame_landmarks.dtype.names:
            add_landmark_point(face_landmarks, frame_landmarks[field_name])
    
    # Add eyes (36-47)
    for i in range(6):
        field_name = f'left_eye_outline_{i}'
        if field_name in frame_landmarks.dtype.names:
            add_landmark_point(face_landmarks, frame_landmarks[field_name])
    for i in range(6):
        field_name = f'right_eye_outline_{i}'
        if field_name in frame_landmarks.dtype.names:
            add_landmark_point(face_landmarks, frame_landmarks[field_name])
    
    # Add lips (48-67)
    for i in range(12):
        field_name = f'outer_lips_{i}'
        if field_name in frame_landmarks.dtype.names:
            add_landmark_point(face_landmarks, frame_landmarks[field_name])
    for i in range(8):
        field_name = f'inner_lips_{i}'
        if field_name in frame_landmarks.dtype.names:
            add_landmark_point(face_landmarks, frame_landmarks[field_name])
    
    # Add remaining facial features in order
    remaining_features = [
        ('pupil_center_', 2),
        ('left_eyebrow_upper_', 5), ('right_eyebrow_upper_', 5),
        ('left_eyebrow_lower_', 5), ('right_eyebrow_lower_', 5),
        ('left_eye_upper0_', 6), ('right_eye_upper0_', 6),
        ('left_eye_lower0_', 6), ('right_eye_lower0_', 6),
        ('left_eye_upper1_', 6), ('right_eye_upper1_', 6),
        ('left_eye_lower1_', 6), ('right_eye_lower1_', 6),
        ('left_eye_upper2_', 6), ('right_eye_upper2_', 6),
        ('left_eye_lower2_', 6), ('right_eye_lower2_', 6)
    ]
    
    for prefix, count in remaining_features:
        for i in range(count):
            field_name = f'{prefix}{i}'
            if field_name in frame_landmarks.dtype.names:
                add_landmark_point(face_landmarks, frame_landmarks[field_name])
    
    return face_landmarks if len(face_landmarks.landmark) > 0 else None

def create_landmark_visualization(video_path, landmarks_npz_path, output_path, gpu_id=0):
    """
    Create a video with landmarks visualized
    
    Args:
        video_path: Path to the original video
        landmarks_npz_path: Path to the NPZ file containing landmarks
        output_path: Path where the output video will be saved
        gpu_id: GPU device ID to use for PyTorch/TensorFlow operations
    """
    # Setup GPU for PyTorch/TensorFlow operations
    gpu_available = setup_gpu(gpu_id)
    
    # Load landmarks data
    print("Loading landmarks data...")
    landmarks_data = np.load(landmarks_npz_path, allow_pickle=True)
    print("Available keys in landmarks file:", landmarks_data.files)
    
    # Get the actual landmarks array - assuming it's stored under 'frame_data'
    landmarks_array = landmarks_data['frame_data']
    print(f"Landmarks array shape: {landmarks_array.shape}")
    print(f"Available fields: {landmarks_array.dtype.names}")
    
    # Initialize MediaPipe solutions
    holistic = mp_holistic.Holistic()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get landmarks for current frame
        if frame_count < len(landmarks_array):
            frame_landmarks = landmarks_array[frame_count]
            print(f"Processing frame {frame_count}, landmark type: {type(frame_landmarks)}")
            
            # Convert landmark coordinates to MediaPipe format
            try:
                # Create landmarks in the correct order
                face_landmarks = create_face_landmarks_from_frame(frame_landmarks)
                pose_landmarks = create_landmark_list()
                
                # Add pose landmarks in order
                pose_fields = [
                    'pose_nose',              # 0
                    'pose_left_eye_inner',    # 1
                    'pose_left_eye',          # 2
                    'pose_left_eye_outer',    # 3
                    'pose_right_eye_inner',   # 4
                    'pose_right_eye',         # 5
                    'pose_right_eye_outer',   # 6
                    'pose_left_ear',          # 7
                    'pose_right_ear',         # 8
                    'pose_mouth_left',        # 9
                    'pose_mouth_right',       # 10
                    'pose_left_shoulder',     # 11
                    'pose_right_shoulder',    # 12
                    'pose_left_elbow',        # 13
                    'pose_right_elbow',       # 14
                    'pose_left_wrist',        # 15
                    'pose_right_wrist',       # 16
                    'pose_left_pinky',        # 17
                    'pose_right_pinky',       # 18
                    'pose_left_index',        # 19
                    'pose_right_index',       # 20
                    'pose_left_thumb',        # 21
                    'pose_right_thumb',       # 22
                    'pose_left_hip',          # 23
                    'pose_right_hip',         # 24
                    'pose_left_knee',         # 25
                    'pose_right_knee',        # 26
                    'pose_left_ankle',        # 27
                    'pose_right_ankle',       # 28
                    'pose_left_heel',         # 29
                    'pose_right_heel',        # 30
                    'pose_left_foot_index',   # 31
                    'pose_right_foot_index'   # 32
                ]

                for field in pose_fields:
                    if field in frame_landmarks.dtype.names:
                        add_landmark_point(pose_landmarks, frame_landmarks[field])
                
                pose_landmarks = pose_landmarks if len(pose_landmarks.landmark) > 0 else None
                
                # Create hand landmarks
                left_hand_landmarks = create_landmark_list_from_points(frame_landmarks['left_hand'])
                right_hand_landmarks = create_landmark_list_from_points(frame_landmarks['right_hand'])

                if all(x is None for x in [face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks]):
                    print(f"Warning: No landmarks found for frame {frame_count}")
                else:
                    print(f"Successfully created landmarks for frame {frame_count}")
                
            except (KeyError, IndexError, AttributeError) as e:
                print(f"Warning: Error processing landmarks for frame {frame_count}: {e}")
                face_landmarks = None
                pose_landmarks = None
                left_hand_landmarks = None
                right_hand_landmarks = None
            
            # Draw landmarks on frame
            draw_landmarks_on_frame(frame, face_landmarks, pose_landmarks, 
                                 left_hand_landmarks, right_hand_landmarks)
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Display progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Release resources
    if gpu_available:
        torch.cuda.empty_cache()
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize landmarks on video')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use for PyTorch/TensorFlow operations')
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to the script directory
    video_path = os.path.join(script_dir, "videos", "1-Introduction-SD.mov")
    landmarks_path = os.path.join(script_dir, "numpy_data", "1-Introduction-SD_landmarks.npz")
    output_path = os.path.join(script_dir, "recreated_videos", "recreated_1-Introduction-SD.mp4")
    
    # Create the visualization
    create_landmark_visualization(video_path, landmarks_path, output_path, args.gpu) 