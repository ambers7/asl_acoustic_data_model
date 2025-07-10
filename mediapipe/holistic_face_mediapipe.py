import cv2
import csv
import os
import mediapipe as mp
import multiprocessing as mp_proc
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import tensorflow as tf

def select_gpu(gpu_id=0):
    """
    Configure the GPU to use for processing.
    Args:
        gpu_id: ID of the GPU to use (default: 0 for first GPU)
    Returns:
        bool: True if GPU was successfully selected
    """
    try:
        # List available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU devices found. Running on CPU.")
            return False

        # Print available GPUs
        print("\nAvailable GPUs:")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")

        if gpu_id >= len(gpus):
            print(f"Warning: GPU {gpu_id} not found. Using GPU 0 instead.")
            gpu_id = 0

        # Configure TensorFlow to use specific GPU
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        
        # Also set CUDA device for OpenCV operations
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        print(f"\nUsing GPU {gpu_id}: {gpus[gpu_id].name}")
        return True

    except Exception as e:
        print(f"Error setting GPU: {e}")
        print("Falling back to CPU")
        return False

# === Paths ===
input_video_path = "videos/1-Introduction-SD.mov"
output_video_path = "output_vids/holisticwithface.mp4"
output_csv_path = "mediapipe_csvs/holistic_withface_landmarks.csv"

# === Setup folders ===
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# === MediaPipe setup ===
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Configure GPU and acceleration
gpu_available = select_gpu(0)  # Try to use first GPU (change number to use different GPU)
if gpu_available:
    cv2.ocl.setUseOpenCL(True)  # Enable OpenCL acceleration
    print("OpenCL acceleration enabled")
else:
    print("Using CPU only mode")

# Optimize CPU threading regardless of GPU status
cv2.setNumThreads(mp_proc.cpu_count())

def list_available_gpus():
    """List all available CUDA GPUs"""
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_count == 0:
            print("No CUDA-capable GPUs found")
            return
        
        print(f"\nFound {gpu_count} CUDA-capable GPU(s):")
        for i in range(gpu_count):
            cv2.cuda.setDevice(i)
            device_name = cv2.cuda.getDevice()
            print(f"GPU {i}: {device_name}")
    except Exception as e:
        print(f"Error listing GPUs: {e}")

def create_holistic():
    """Create a holistic instance with optimized settings"""
    return mp_holistic.Holistic(
        static_image_mode=False,  # Faster for video
        model_complexity=1,  # Balance between speed and accuracy (0=fastest, 2=most accurate)
        smooth_landmarks=True,
        enable_segmentation=False,  # Disable unused features
        refine_face_landmarks=True,
        min_detection_confidence=0.5,  # Lower threshold for faster detection
        min_tracking_confidence=0.5
    )

def process_frame(frame_data):
    """Process a single frame with holistic detection"""
    frame_idx, frame = frame_data
    
    # Create holistic instance for this thread
    with mp_holistic.Holistic(**holistic_config) as holistic:
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = holistic.process(rgb)
        
        # Draw landmarks
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_styles.get_default_pose_landmarks_style())

        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style())

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style())
        
        # Extract landmarks
        landmarks_data = []
        
        def extract_landmarks(landmarks, kind):
            if landmarks:
                for idx, lm in enumerate(landmarks.landmark):
                    landmarks_data.append([
                        frame_idx, kind, idx, lm.x, lm.y, lm.z,
                        getattr(lm, 'visibility', '')
                    ])
        
        extract_landmarks(results.pose_landmarks, "pose")
        extract_landmarks(results.face_landmarks, "face")
        extract_landmarks(results.left_hand_landmarks, "left_hand")
        extract_landmarks(results.right_hand_landmarks, "right_hand")
        
        return frame_idx, annotated_frame, landmarks_data

def main():
    # === Video input/output ===
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # === CSV setup ===
    csv_file = open(output_csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "type", "landmark_index", "x", "y", "z", "visibility"])
    
    # === Process frames in batches ===
    batch_size = 32  # Adjust based on available memory
    
    try:
        # parallel processing
        with ThreadPoolExecutor(max_workers=mp_proc.cpu_count()) as executor:
            frame_idx = 0
            while frame_idx < total_frames:
                # Read batch of frames
                frames_batch = []
                for _ in range(batch_size):
                    if frame_idx >= total_frames:
                        break
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frames_batch.append((frame_idx, frame))
                    frame_idx += 1  # Process every frame (no skipping)
                
                if not frames_batch:
                    break
                
                # Process batch in parallel
                futures = [executor.submit(process_frame, frame_data) for frame_data in frames_batch]
                
                # Collect results
                results = []
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                
                # Sort results by frame index
                results.sort(key=lambda x: x[0])
                
                # Write results
                for frame_idx, annotated_frame, landmarks_data in results:
                    # Write landmarks to CSV
                    for row in landmarks_data:
                        csv_writer.writerow(row)
                    
                    # Write frame to video
                    out.write(annotated_frame)
                
                print(f"Processed frames up to {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%)")
    
    finally:
        # === Cleanup ===
        cap.release()
        out.release()
        csv_file.close()
        print("✅ Done! Video and CSV saved.")

if __name__ == '__main__':
    # Configuration for holistic instances
    holistic_config = {
        'static_image_mode': False,  # Faster for video
        'model_complexity': 1,       # Balance speed/accuracy
        'smooth_landmarks': True,    # Smoother tracking
        'enable_segmentation': False,# Disable unused features
        'refine_face_landmarks': True,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    }
    
    main()
