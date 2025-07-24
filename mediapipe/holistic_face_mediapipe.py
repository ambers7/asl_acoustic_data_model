# Basic logging configuration - keep only what's necessary
import os
import logging
import warnings
import csv
import sys
import mediapipe as mp
import multiprocessing as mp_proc
from concurrent.futures import ThreadPoolExecutor, as_completed
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
input_video_path = "mediapipe/videos/1-Introduction-SD.mov"
output_video_path = "mediapipe/output_vids/everythingholistic.mp4"
output_csv_path = "mediapipe/mediapipe_csvs/everythingholistic.csv"

# === Setup folders ===
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

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

# Define pose landmark names
POSE_LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index"
}

# Define hand landmark names
HAND_LANDMARK_NAMES = {
    0: "wrist",
    1: "thumb_cmc",
    2: "thumb_mcp",
    3: "thumb_ip",
    4: "thumb_tip",
    5: "index_mcp",
    6: "index_pip",
    7: "index_dip",
    8: "index_tip",
    9: "middle_mcp",
    10: "middle_pip",
    11: "middle_dip",
    12: "middle_tip",
    13: "ring_mcp",
    14: "ring_pip",
    15: "ring_dip",
    16: "ring_tip",
    17: "pinky_mcp",
    18: "pinky_pip",
    19: "pinky_dip",
    20: "pinky_tip"
}

# Define face landmark regions
FACE_LANDMARK_REGIONS = {
    # Face oval (silhouette)
    range(0, 17): "face_oval",
    
    # Eyebrows
    range(17, 22): "left_eyebrow",
    range(22, 27): "right_eyebrow",
    range(70, 75): "left_eyebrow_upper",
    range(75, 80): "right_eyebrow_upper",
    range(80, 85): "left_eyebrow_lower",
    range(85, 90): "right_eyebrow_lower",
    
    # Eyes detailed
    range(27, 31): "nose_bridge",
    range(31, 36): "nose_tip",
    range(36, 42): "left_eye_outline",
    range(42, 48): "right_eye_outline",
    range(48, 60): "outer_lips",
    range(60, 68): "inner_lips",
    range(68, 70): "pupil_center",
    
    # Eyes additional detail
    range(90, 96): "left_eye_upper0",
    range(96, 102): "right_eye_upper0",
    range(102, 108): "left_eye_lower0",
    range(108, 114): "right_eye_lower0",
    range(114, 120): "left_eye_upper1",
    range(120, 126): "right_eye_upper1",
    range(126, 132): "left_eye_lower1",
    range(132, 138): "right_eye_lower1",
    range(138, 144): "left_eye_upper2",
    range(144, 150): "right_eye_upper2",
    range(150, 156): "left_eye_lower2",
    range(156, 162): "right_eye_lower2",
    
    # Nose detailed
    range(162, 168): "nose_bridge_detailed",
    range(168, 174): "nose_tip_detailed",
    range(174, 180): "nose_bottom",
    range(180, 186): "nose_right_outline",
    range(186, 192): "nose_left_outline",
    
    # Lips detailed
    range(192, 204): "upper_lip_top",
    range(204, 216): "upper_lip_bottom",
    range(216, 228): "lower_lip_top",
    range(228, 240): "lower_lip_bottom",
    
    # Mouth cavity
    range(240, 252): "mouth_cavity_upper",
    range(252, 264): "mouth_cavity_lower",
    
    # Face cheeks and additional contours
    range(264, 270): "left_cheek_upper",
    range(270, 276): "right_cheek_upper",
    range(276, 282): "left_cheek_lower",
    range(282, 288): "right_cheek_lower",
    
    # Face additional contours
    range(288, 300): "face_contour_upper",
    range(300, 312): "face_contour_lower",
    range(312, 324): "face_contour_cheeks",
    
    # Detailed eye regions
    range(324, 336): "left_eye_creases",
    range(336, 348): "right_eye_creases",
    range(348, 360): "left_eye_wrinkles",
    range(360, 372): "right_eye_wrinkles",
    
    # Additional facial features
    range(372, 384): "forehead_upper",
    range(384, 396): "forehead_lower",
    range(396, 408): "temple_left",
    range(408, 420): "temple_right",
    
    # Detailed nose features
    range(420, 432): "nose_bridge_wrinkles",
    range(432, 444): "nose_side_wrinkles",
    range(444, 456): "nose_tip_detailed_wrinkles",
    
    # Additional mouth features
    range(456, 468): "mouth_corners_detailed",
    
    # Iris landmarks (if available)
    range(468, 469): "left_iris",
    range(469, 470): "right_iris",
    
    # Additional face mesh points
    range(470, 474): "left_eye_mesh_extra",
    range(474, 478): "right_eye_mesh_extra"
}

def get_face_landmark_name(idx):
    """Get the region name for a face landmark index"""
    for range_key, region in FACE_LANDMARK_REGIONS.items():
        if idx in range_key:
            point_idx = idx - min(range_key)
            # For single point regions, don't add the index
            if len(range_key) == 1:
                return region
            return f"{region}_{point_idx}"
    return f"face_point_{idx}"

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

def create_new_csv_writer(base_path, part_num):
    """Create a new CSV writer with the given part number"""
    # Create a folder based on video name
    video_name = get_video_name()
    csv_dir = os.path.dirname(base_path)
    video_specific_dir = os.path.join(csv_dir, video_name)
    os.makedirs(video_specific_dir, exist_ok=True)
    
    # Create the new path inside the video-specific directory
    base_filename = f"landmarks_part{part_num}.csv"
    new_path = os.path.join(video_specific_dir, base_filename)
    
    csv_file = open(new_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "type", "landmark_index", 
                    "x_norm", "y_norm", "z_norm",  # Normalized coordinates
                    "x_px", "y_px",  # Pixel coordinates
                    "visibility"])
    return csv_file, csv_writer, new_path

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

def clear_gpu_memory():
    """Clear GPU memory between batches"""
    if gpu_available:
        torch.cuda.empty_cache()  # If PyTorch is available
        gc.collect()  # Force garbage collection
        tf.keras.backend.clear_session()  # Clear TF session

# High Quality Settings (More Accurate but Slower and More Memory)
"""
HOLISTIC_INSTANCE = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,  # Use most complex model for best accuracy
    smooth_landmarks=True,
    enable_segmentation=True,  # Enable background segmentation
    refine_face_landmarks=True,  # Use additional refinement for face
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
"""

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

# What each setting affects:
# static_image_mode=False 
#   - Optimizes for video by using tracking between frames
#   - Faster than processing each frame independently

# model_complexity=0
#   - Uses lightweight neural networks
#   - Reduces RAM and VRAM usage significantly
#   - May miss subtle movements but catches main poses

# smooth_landmarks=True
#   - Reduces jitter in landmark positions
#   - Minimal performance impact, big quality benefit

# enable_segmentation=False
#   - Disables person/background separation
#   - Saves significant GPU memory
#   - Not needed for landmark tracking

# refine_face_landmarks=False
#   - Skips additional face landmark refinement
#   - Still gets all 468 face points
#   - Points might be slightly less precise around eyes/lips
#   - Saves GPU memory and processing time

# min_detection_confidence=0.5
# min_tracking_confidence=0.5
#   - Default values work well for most cases
#   - Lower values = more detections but more false positives
#   - Higher values = fewer detections but more accurate

def process_frame(frame_data):
    """Process a single frame with holistic detection"""
    global width, height
    frame_idx, frame = frame_data
    
    try:
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = HOLISTIC_INSTANCE.process(rgb)
        
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
                    # Convert normalized coordinates to pixel coordinates
                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)
                    
                    # Use anatomical names for pose and hand landmarks
                    if kind == "pose":
                        landmark_type = POSE_LANDMARK_NAMES.get(idx, idx)
                    elif kind in ["left_hand", "right_hand"]:
                        hand_name = HAND_LANDMARK_NAMES.get(idx, idx)
                        landmark_type = f"{kind}_{hand_name}"
                    elif kind == "face":
                        landmark_type = get_face_landmark_name(idx)
                    else:
                        landmark_type = kind
                    
                    landmarks_data.append([
                        frame_idx,
                        landmark_type,  # Use anatomical name for landmarks
                        idx,
                        lm.x,  # normalized x
                        lm.y,  # normalized y
                        lm.z,  # normalized z
                        x_px,  # pixel x
                        y_px,  # pixel y
                        getattr(lm, 'visibility', '')
                    ])
        
        extract_landmarks(results.pose_landmarks, "pose")
        extract_landmarks(results.face_landmarks, "face")
        extract_landmarks(results.left_hand_landmarks, "left_hand")
        extract_landmarks(results.right_hand_landmarks, "right_hand")
        
        return frame_idx, annotated_frame, landmarks_data
    except Exception as e:
        print(f"\nError processing frame {frame_idx}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_batch(frames_batch, executor):
    """Process a batch of frames with memory management"""
    try:
        # Process frames sequentially to maintain order
        results = []
        for frame_data in frames_batch:
            try:
                # Clear memory before each frame
                clear_gpu_memory()
                
                # Process directly without using executor
                result = process_frame(frame_data)
                results.append(result)
                
                # Force garbage collection after each frame
                gc.collect()
            except Exception as e:
                print(f"Error processing frame {frame_data[0]}: {str(e)}")
                continue
        
        return results
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return []

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
        
        # === CSV setup ===
        current_part = 1
        frames_per_file = 124  # Keep small file size for stability
        csv_file, csv_writer, current_csv_path = create_new_csv_writer(output_csv_path, current_part)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # === Process frames in small batches ===
        batch_size = 2  # Reduced batch size for stability
        total_landmarks = 0
        last_progress_frame = 0
        error_frames = []
        frames_in_current_file = 0
        last_memory_check = 0
        batch_count = 0
        
        # Sequential processing
        with ThreadPoolExecutor(max_workers=1) as executor:  # Single worker
            frame_idx = 0
            while frame_idx < total_frames:
                try:
                    batch_count += 1
                    
                    # Force garbage collection every batch
                    clear_gpu_memory()
                    memory_usage = get_memory_usage()
                    last_memory_check = frame_idx
                    
                    # Check if we need to start a new CSV file
                    if frames_in_current_file >= frames_per_file:
                        csv_file.close()
                        current_part += 1
                        csv_file, csv_writer, current_csv_path = create_new_csv_writer(output_csv_path, current_part)
                        frames_in_current_file = 0
                        # Force GC after file switch
                        clear_gpu_memory()
                    
                    # Read batch of frames
                    frames_batch = []
                    batch_start_idx = frame_idx
                    frames_read = 0
                    
                    for _ in range(batch_size):
                        if frame_idx >= total_frames:
                            break
                        
                        try:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                            if not ret:
                                print(f"Error: Failed to read frame {frame_idx}")
                                error_frames.append(frame_idx)
                                break
                            
                            frames_batch.append((frame_idx, frame))
                            frame_idx += 1
                            frames_read += 1
                        except Exception as e:
                            print(f"Error reading frame {frame_idx}: {str(e)}")
                            error_frames.append(frame_idx)
                            frame_idx += 1
                            continue
                    
                    if not frames_batch:
                        print("No more frames to process")
                        break
                    
                    # Process batch
                    results = process_batch(frames_batch, executor)
                    
                    if not results:
                        print(f"Warning: No results from batch {batch_start_idx} to {frame_idx-1}")
                        continue
                    
                    # Write results
                    batch_landmarks = 0
                    successful_writes = 0
                    for result_frame_idx, annotated_frame, landmarks_data in results:
                        try:
                            # Write landmarks to CSV
                            for row in landmarks_data:
                                csv_writer.writerow(row)
                                batch_landmarks += 1
                            
                            # Write frame to video
                            out.write(annotated_frame)
                            successful_writes += 1
                        except Exception as e:
                            print(f"Error writing results for frame {result_frame_idx}: {str(e)}")
                            error_frames.append(result_frame_idx)
                            continue
                    
                    total_landmarks += batch_landmarks
                    frames_in_current_file += len(results)
                    
                    # Flush CSV writer every batch
                    csv_file.flush()
                    
                    # Show progress every 10 frames
                    if frame_idx - last_progress_frame >= 10:
                        print(f"\n=== Batch {batch_count} Complete ===")
                        print(f"Progress: {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%)")
                        print(f"Memory usage: {get_memory_usage():.1f} MB")
                        if error_frames:
                            print(f"Frames with errors: {error_frames}")
                        last_progress_frame = frame_idx
                    
                except Exception as e:
                    print(f"Error in main processing loop at frame {frame_idx}: {str(e)}")
                    print("Stack trace:")
                    import traceback
                    traceback.print_exc()
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
        try:
            csv_file.close()
        except:
            pass
        
        print("\n=== Processing Complete ===")
        print(f"Final memory usage: {get_memory_usage():.1f} MB")
        print(f"Processed {frame_idx}/{total_frames} frames")
        print(f"Total landmarks written: {total_landmarks}")
        print(f"Total CSV parts: {current_part}")
        if error_frames:
            print(f"Frames with errors: {sorted(set(error_frames))}")  # Remove duplicates and sort
        print(f"Output files:")
        print(f"- Video: {output_video_path}")
        print("- CSV files:")
        video_name = get_video_name()
        csv_dir = os.path.dirname(output_csv_path)
        video_specific_dir = os.path.join(csv_dir, video_name)
        print(f"  Directory: {video_specific_dir}")
        for part in range(1, current_part + 1):
            print(f"  - landmarks_part{part}.csv")

if __name__ == '__main__':
    main()
