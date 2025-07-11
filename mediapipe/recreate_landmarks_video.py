import cv2
import numpy as np
import mediapipe as mp
import os
import torch
import tensorflow as tf

def setup_gpu(gpu_id=0):
    """Setup GPU configuration"""
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
        return True
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

def preprocess_frame_gpu(frame):
    """Preprocess frame using GPU"""
    if torch.cuda.is_available():
        # Convert frame to GPU tensor
        frame_tensor = torch.from_numpy(frame).cuda()
        # Add batch dimension
        frame_tensor = frame_tensor.unsqueeze(0)
        # Normalize
        frame_tensor = frame_tensor.float() / 255.0
        return frame_tensor
    return frame

def postprocess_frame_gpu(frame_tensor):
    """Postprocess frame tensor back to CPU numpy array"""
    if torch.cuda.is_available():
        # Convert back to numpy
        frame = frame_tensor.squeeze(0).cpu().numpy()
        # Convert back to uint8
        frame = (frame * 255).astype(np.uint8)
        return frame
    return frame

def create_landmark_visualization(video_path, landmarks_npz_path, output_path, gpu_id=0):
    """
    Create a video with landmarks visualized using GPU acceleration
    
    Args:
        video_path: Path to the original video
        landmarks_npz_path: Path to the NPZ file containing landmarks
        output_path: Path where the output video will be saved
        gpu_id: GPU device ID to use
    """
    # Setup GPU
    gpu_available = setup_gpu(gpu_id)
    if gpu_available:
        print(f"Using GPU {gpu_id}")
        # Set OpenCV to use CUDA backend
        cv2.setUseOptimized(True)
        cv2.cuda.setDevice(gpu_id)
    else:
        print("GPU not available, using CPU")
    
    # Load landmarks data
    print("Loading landmarks data...")
    landmarks_data = np.load(landmarks_npz_path, allow_pickle=True)
    
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
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create CUDA stream if GPU is available
    if gpu_available:
        stream = cv2.cuda_Stream()
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if gpu_available:
            # Upload frame to GPU
            frame_gpu = preprocess_frame_gpu(frame)
        
        # Get landmarks for current frame
        if frame_count < len(landmarks_data):
            frame_landmarks = landmarks_data[frame_count]
            
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
            if gpu_available:
                frame = postprocess_frame_gpu(frame_gpu)
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
    parser = argparse.ArgumentParser(description='Visualize landmarks on video using GPU acceleration')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--landmarks', required=True, help='Path to landmarks NPZ file')
    parser.add_argument('--output', required=True, help='Path for output video')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
    
    args = parser.parse_args()
    
    create_landmark_visualization(args.video, args.landmarks, args.output, args.gpu) 