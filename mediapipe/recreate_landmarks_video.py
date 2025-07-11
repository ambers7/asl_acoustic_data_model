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
    
    # Get the actual landmarks array - assuming it's stored under 'landmarks' or similar key
    if 'landmarks' in landmarks_data.files:
        landmarks_array = landmarks_data['landmarks']
    else:
        # If there's only one array, take the first key
        landmarks_array = landmarks_data[landmarks_data.files[0]]
    
    print(f"Landmarks array shape: {landmarks_array.shape}")
    
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
            
            # Convert landmark coordinates to MediaPipe format
            face_landmarks = mp_holistic.face_landmarks.FaceLandmark(
                x=frame_landmarks['face_landmarks'][0],
                y=frame_landmarks['face_landmarks'][1],
                z=frame_landmarks['face_landmarks'][2]
            ) if 'face_landmarks' in frame_landmarks else None
            
            pose_landmarks = mp_holistic.pose_landmarks.PoseLandmark(
                x=frame_landmarks['pose_landmarks'][0],
                y=frame_landmarks['pose_landmarks'][1],
                z=frame_landmarks['pose_landmarks'][2]
            ) if 'pose_landmarks' in frame_landmarks else None
            
            left_hand_landmarks = mp_holistic.hand_landmarks.HandLandmark(
                x=frame_landmarks['left_hand_landmarks'][0],
                y=frame_landmarks['left_hand_landmarks'][1],
                z=frame_landmarks['left_hand_landmarks'][2]
            ) if 'left_hand_landmarks' in frame_landmarks else None
            
            right_hand_landmarks = mp_holistic.hand_landmarks.HandLandmark(
                x=frame_landmarks['right_hand_landmarks'][0],
                y=frame_landmarks['right_hand_landmarks'][1],
                z=frame_landmarks['right_hand_landmarks'][2]
            ) if 'right_hand_landmarks' in frame_landmarks else None
            
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