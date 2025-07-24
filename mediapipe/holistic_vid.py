# Basic logging configuration - keep only what's necessary
import os
import logging
import warnings
import sys
import mediapipe as mp
import multiprocessing as mp_proc
import numpy as np
import tensorflow as tf
import psutil
import gc
import cv2
import torch # Added for PyTorch GPU memory management

# Configure GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging

# For better GPU memory management
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent TF from taking all GPU memory
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Use async memory allocator

# Basic logging setup
logging.getLogger().setLevel(logging.ERROR)

# === Paths ===
input_video_path = "mediapipe/videos/laptop_cam.mp4"
output_video_path = "mediapipe/output_vids/laptop_cam_landmarks.mp4"

# === Setup folders ===
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# Custom warning filter for MediaPipe
def mediapipe_warning_filter(record):
    message = str(record.msg).lower()
    return not any(x in message for x in [
        'inference_feedback_manager',
        'feedback tensors',
        'feedback manager',
        'gl_context',
        'successfully initialized',
        'gl version',
        'renderer'
    ])

# Apply more filters
logging.getLogger('mediapipe').addFilter(mediapipe_warning_filter)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Disable TF debugging and GPU initialization messages
tf.debugging.disable_traceback_filtering()
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# === Global variables ===
width = 0
height = 0

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

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def get_video_name():
    """Extract video name from input path without extension"""
    video_basename = os.path.basename(input_video_path)
    return os.path.splitext(video_basename)[0]

def clear_gpu_memory():
    """Clear GPU memory between batches"""
    if gpu_available:
        torch.cuda.empty_cache()  # If PyTorch is available
        gc.collect()  # Force garbage collection
        tf.keras.backend.clear_session()  # Clear TF session

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

# Current Performance Settings (Faster and Less Memory)
HOLISTIC_INSTANCE = mp_holistic.Holistic(
    static_image_mode=True,  # Process frames independently
    model_complexity=0,  # Use simplest model
    smooth_landmarks=False,  # Disable temporal smoothing
    enable_segmentation=False,  # Disable segmentation
    refine_face_landmarks=False,  # Skip face refinement
    min_detection_confidence=0.3,  # Lower threshold for faster processing
    min_tracking_confidence=0.3  # Lower threshold since we're not tracking
)

def process_frame(frame):
    """Process a single frame with holistic detection"""
    try:
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = HOLISTIC_INSTANCE.process(rgb)
        
        # Draw landmarks
        annotated_frame = frame.copy()

        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_styles.get_default_pose_landmarks_style())

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style())

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style())
        
        return annotated_frame
    except Exception as e:
        print(f"\nError processing frame: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    global width, height
    try:
        print("\n=== Starting Video Processing ===")
        print(f"Initial memory usage: {get_memory_usage():.1f} MB")
        
        # === Video input/output ===
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {input_video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video: {input_video_path}")
        print(f"Video name: {get_video_name()}")
        print(f"Total frames: {total_frames}")
        print(f"Video dimensions: {width}x{height}")
        print(f"FPS: {fps}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        last_progress_frame = 0
        error_frames = []
        
        while True:
            try:
                # Force garbage collection periodically
                if frame_idx % 10 == 0:
                    clear_gpu_memory()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame = process_frame(frame)
                if annotated_frame is not None:
                    out.write(annotated_frame)
                else:
                    error_frames.append(frame_idx)
                
                frame_idx += 1
                
                # Show progress every 10 frames
                if frame_idx - last_progress_frame >= 10:
                    print(f"\nProgress: {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%)")
                    print(f"Memory usage: {get_memory_usage():.1f} MB")
                    if error_frames:
                        print(f"Frames with errors: {error_frames}")
                    last_progress_frame = frame_idx
                    
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                error_frames.append(frame_idx)
                frame_idx += 1
                continue
    
    except Exception as e:
        print(f"\nFatal error in main processing loop: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # === Cleanup ===
        try:
            HOLISTIC_INSTANCE.close()  # Clean up the holistic instance
        except:
            pass
        try:
            cap.release()
        except:
            pass
        try:
            out.release()
        except:
            pass
        
        print("\n=== Processing Complete ===")
        print(f"Final memory usage: {get_memory_usage():.1f} MB")
        print(f"Processed {frame_idx}/{total_frames} frames")
        if error_frames:
            print(f"Frames with errors: {sorted(set(error_frames))}")  # Remove duplicates and sort
        print(f"Output video: {output_video_path}")

if __name__ == '__main__':
    main()
