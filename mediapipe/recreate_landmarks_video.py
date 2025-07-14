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
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        # Draw face contours
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
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

def create_landmark_visualization(video_path, landmarks_npz_path, output_path, gpu_id=0):
    """
    Create a video with landmarks visualized
    
    Args:
        video_path: Path to the original video
        landmarks_npz_path: Path to the NPZ file containing landmarks
        output_path: Path where the output video will be saved
        gpu_id: GPU device ID to use (for PyTorch/TensorFlow operations)
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
                # Create proper landmark objects
                def create_landmark_list(prefix, num_landmarks):
                    landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                    for i in range(num_landmarks):
                        field_name = f"{prefix}_{i}"
                        if field_name in frame_landmarks.dtype.names and not np.all(frame_landmarks[field_name] == 0):
                            landmark = landmark_list.landmark.add()
                            landmark.x = float(frame_landmarks[field_name][0])
                            landmark.y = float(frame_landmarks[field_name][1])
                            landmark.z = float(frame_landmarks[field_name][2])
                            landmark.visibility = float(frame_landmarks[field_name][3])
                    return landmark_list if len(landmark_list.landmark) > 0 else None

                def create_hand_landmark_list(hand_data):
                    if np.all(hand_data == 0):
                        return None
                    landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                    for point in hand_data:
                        if not np.all(point == 0):
                            landmark = landmark_list.landmark.add()
                            landmark.x = float(point[0])
                            landmark.y = float(point[1])
                            landmark.z = float(point[2])
                            landmark.visibility = float(point[3])
                    return landmark_list if len(landmark_list.landmark) > 0 else None

                # Create face landmarks from all face-related fields
                face_landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                
                # List of all facial feature prefixes from save_landmarks_numpy.py
                facial_features = [
                    'face_oval_',           # Face silhouette
                    'left_eyebrow_', 'right_eyebrow_',  # Basic eyebrows
                    'left_eyebrow_upper_', 'right_eyebrow_upper_',  # Upper eyebrows
                    'left_eyebrow_lower_', 'right_eyebrow_lower_',  # Lower eyebrows
                    'nose_bridge_', 'nose_tip_',  # Basic nose
                    'left_eye_outline_', 'right_eye_outline_',  # Basic eye outlines
                    'outer_lips_', 'inner_lips_',  # Basic lips
                    'pupil_center_',  # Pupils
                    'left_eye_upper0_', 'right_eye_upper0_',  # Detailed eyes
                    'left_eye_lower0_', 'right_eye_lower0_',
                    'left_eye_upper1_', 'right_eye_upper1_',
                    'left_eye_lower1_', 'right_eye_lower1_',
                    'left_eye_upper2_', 'right_eye_upper2_',
                    'left_eye_lower2_', 'right_eye_lower2_',
                    'nose_bridge_detailed_', 'nose_tip_detailed_',  # Detailed nose
                    'nose_bottom_', 'nose_right_outline_', 'nose_left_outline_',
                    'upper_lip_top_', 'upper_lip_bottom_',  # Detailed lips
                    'lower_lip_top_', 'lower_lip_bottom_',
                    'mouth_cavity_upper_', 'mouth_cavity_lower_',  # Mouth cavity
                    'left_cheek_upper_', 'right_cheek_upper_',  # Cheeks
                    'left_cheek_lower_', 'right_cheek_lower_',
                    'face_contour_upper_', 'face_contour_lower_',  # Face contours
                    'face_contour_cheeks_',
                    'left_eye_creases_', 'right_eye_creases_',  # Eye details
                    'left_eye_wrinkles_', 'right_eye_wrinkles_',
                    'forehead_upper_', 'forehead_lower_',  # Forehead
                    'temple_left_', 'temple_right_',  # Temples
                    'nose_bridge_wrinkles_', 'nose_side_wrinkles_',  # Nose details
                    'nose_tip_detailed_wrinkles_',
                    'mouth_corners_detailed_',  # Mouth details
                    'left_iris', 'right_iris',  # Iris
                    'left_eye_mesh_extra_', 'right_eye_mesh_extra_'  # Extra eye mesh points
                ]

                # Add landmarks for each facial feature
                for prefix in facial_features:
                    # For single-point features (like iris)
                    if prefix in ['left_iris', 'right_iris']:
                        if prefix in frame_landmarks.dtype.names and not np.all(frame_landmarks[prefix] == 0):
                            landmark = face_landmarks.landmark.add()
                            landmark.x = float(frame_landmarks[prefix][0])
                            landmark.y = float(frame_landmarks[prefix][1])
                            landmark.z = float(frame_landmarks[prefix][2])
                            landmark.visibility = float(frame_landmarks[prefix][3])
                        continue

                    # For multi-point features
                    max_points = 20  # Large enough number to cover all possible points
                    for i in range(max_points):
                        field_name = f"{prefix}{i}"
                        if field_name in frame_landmarks.dtype.names and not np.all(frame_landmarks[field_name] == 0):
                            landmark = face_landmarks.landmark.add()
                            landmark.x = float(frame_landmarks[field_name][0])
                            landmark.y = float(frame_landmarks[field_name][1])
                            landmark.z = float(frame_landmarks[field_name][2])
                            landmark.visibility = float(frame_landmarks[field_name][3])

                face_landmarks = face_landmarks if len(face_landmarks.landmark) > 0 else None

                # Create pose landmarks
                pose_landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
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
                    if field in frame_landmarks.dtype.names and not np.all(frame_landmarks[field] == 0):
                        landmark = pose_landmarks.landmark.add()
                        landmark.x = float(frame_landmarks[field][0])
                        landmark.y = float(frame_landmarks[field][1])
                        landmark.z = float(frame_landmarks[field][2])
                        landmark.visibility = float(frame_landmarks[field][3])

                pose_landmarks = pose_landmarks if len(pose_landmarks.landmark) > 0 else None

                # Create hand landmarks
                left_hand_landmarks = create_hand_landmark_list(frame_landmarks['left_hand'])
                right_hand_landmarks = create_hand_landmark_list(frame_landmarks['right_hand'])

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