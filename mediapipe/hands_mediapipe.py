import cv2
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# === Step 1: Set up HandLandmarker ===
base_options = python.BaseOptions(model_asset_path='mediapipe_models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.HandLandmarker.create_from_options(options)

# === Step 2: Video I/O ===
input_video = "videos/1-Introduction-SD.mov"
output_video = "output_vids/annotated_hands.mp4"
output_csv = "mediapipe_csvs/hand_landmarks.csv"

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_idx = 0

out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

# === Step 3: Setup CSV writer ===
csv_file = open(output_csv, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "hand_index", "handedness", "landmark_index", "x", "y", "z"])

# === Step 4: Draw landmarks function ===
def draw_hand_landmarks(frame_rgb, detection_result):
    annotated_image = np.copy(frame_rgb)
    for hand_landmarks in detection_result.hand_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=proto,
            connections=mp.solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style()
        )
    return annotated_image

# === Step 5: Frame-by-frame processing ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((frame_idx / fps) * 1000)

    result = detector.detect_for_video(mp_image, timestamp_ms)

    # === Save landmarks to CSV ===
    if result.hand_landmarks and result.handedness:
        for hand_idx, (hand_landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
            label = handedness[0].category_name  # "Left" or "Right"
            for lm_idx, lm in enumerate(hand_landmarks):
                csv_writer.writerow([
                    frame_idx,
                    hand_idx,
                    label,
                    lm_idx,
                    lm.x,
                    lm.y,
                    lm.z
                ])

    # === Draw and write annotated frame ===
    annotated = draw_hand_landmarks(frame_rgb, result)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    out.write(annotated_bgr)

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Processed frame {frame_idx}")

# === Cleanup ===
cap.release()
out.release()
csv_file.close()
print(f"✅ Done! Output video: {output_video}\n📄 Landmark CSV: {output_csv}")
