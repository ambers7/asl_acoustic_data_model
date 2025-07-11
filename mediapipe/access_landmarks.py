import numpy as np
import os

def load_landmarks(npz_path):
    """Load landmarks from a saved .npz file"""
    # Load with allow_pickle=True and a larger max_header_size
    data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
    return data['frame_data']

def get_landmark_groups(frame_data):
    """Get all available landmark groups in the data"""
    groups = {
        'pose': [name for name in frame_data.dtype.names if name.startswith('pose_')],
        'face_oval': [name for name in frame_data.dtype.names if name.startswith('face_oval_')],
        'eyebrows': [name for name in frame_data.dtype.names if any(name.startswith(p) for p in ['left_eyebrow_', 'right_eyebrow_'])],
        'eyes': [name for name in frame_data.dtype.names if any(name.startswith(p) for p in ['left_eye_', 'right_eye_'])],
        'nose': [name for name in frame_data.dtype.names if name.startswith('nose_')],
        'mouth': [name for name in frame_data.dtype.names if any(name.startswith(p) for p in ['mouth_', 'upper_lip_', 'lower_lip_'])],
        'cheeks': [name for name in frame_data.dtype.names if any(name.startswith(p) for p in ['left_cheek_', 'right_cheek_'])],
        'hands': ['left_hand', 'right_hand']
    }
    return groups

def get_frame_landmarks(frame_data, frame_idx):
    """Get all landmarks for a specific frame"""
    return frame_data[frame_idx]

def get_hand_landmarks(frame, hand='left'):
    """Get hand landmarks for a specific frame
    Returns array of shape (21, 4) where each row is [x, y, z, visibility]
    """
    hand_key = f'{hand}_hand'
    return frame[hand_key]

def get_face_region_landmarks(frame, region_prefix):
    """Get all landmarks for a specific face region
    Example region_prefixes: 'face_oval_', 'left_eye_', 'nose_', etc.
    """
    region_fields = [name for name in frame.dtype.names if name.startswith(region_prefix)]
    landmarks = {}
    for field in region_fields:
        landmarks[field] = frame[field]
    return landmarks

def print_landmark_info(frame, landmark_name):
    """Print information about a specific landmark"""
    if landmark_name in frame.dtype.names:
        data = frame[landmark_name]
        print(f"\n{landmark_name}:")
        print(f"x: {data[0]:.4f}")
        print(f"y: {data[1]:.4f}")
        print(f"z: {data[2]:.4f}")
        print(f"visibility: {data[3]:.4f}")
    else:
        print(f"Landmark {landmark_name} not found")

def main():
    # Example usage
    npz_path = "mediapipe/numpy_data/1-Introduction-SD_landmarks.npz"
    
    if not os.path.exists(npz_path):
        print(f"Error: File not found: {npz_path}")
        return
    
    try:
        # Load the data
        print("\nLoading landmark data...")
        frame_data = load_landmarks(npz_path)
        print(f"Loaded {len(frame_data)} frames")
        
        # Get available landmark groups
        groups = get_landmark_groups(frame_data)
        print("\nAvailable landmark groups:")
        for group, landmarks in groups.items():
            print(f"{group}: {len(landmarks)} landmarks")
        
        # Example: Get landmarks for frame 110
        print("\nAccessing frame 126...")
        frame = get_frame_landmarks(frame_data, 126)
        
        # Example: Print some basic pose landmarks
        print("\nPose landmarks:")
        print_landmark_info(frame, 'pose_nose')
        print_landmark_info(frame, 'pose_left_eye')
        print_landmark_info(frame, 'pose_right_eye')
        
        # Example: Get all face oval landmarks
        face_oval = get_face_region_landmarks(frame, 'face_oval_')
        print(f"\nFace oval landmarks: {len(face_oval)} points")
        
        # Example: Get hand landmarks
        left_hand = get_hand_landmarks(frame, 'left')
        right_hand = get_hand_landmarks(frame, 'right')
        print("\nHand landmarks:")
        print(f"Left hand: {len(left_hand)} points")
        print(f"Right hand: {len(right_hand)} points")
        
        # Example: Get all eye-related landmarks
        eye_landmarks = get_face_region_landmarks(frame, 'left_eye_')
        print(f"\nLeft eye landmarks: {len(eye_landmarks)} points")
        
        # Example: Print utterance data if available
        print("\nUtterance data:")
        print(f"#: {frame['file_number']}")
        print(f"Utterance ID: {frame['utterance_id']}")
        print(f"Manual signs: {frame['manual_signs']}")
        print(f"Non-manual signs: {frame['non_manual_signs']}")
        
    except Exception as e:
        print(f"Error loading or processing landmarks: {e}")

if __name__ == '__main__':
    main() 