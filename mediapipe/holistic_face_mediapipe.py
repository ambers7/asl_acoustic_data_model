import cv2
import csv
import os
import mediapipe as mp

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

holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=1,
                                smooth_landmarks=True,
                                enable_segmentation=False,
                                refine_face_landmarks=True)

# === Video input/output ===
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# === CSV setup ===
csv_file = open(output_csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "type", "landmark_index", "x", "y", "z", "visibility"])

# === Process each frame ===
frame_idx = 0
frame_skip = 10  # ← change this to 2, 5, 10 depending on desired speed/accuracy


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if frame_idx % frame_skip != 0:
        frame_idx += 1
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    # --- Draw landmarks ---
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_styles.get_default_pose_landmarks_style())

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style())

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style())

    # --- Write to CSV ---
    def write_landmarks(landmarks, kind):
        for idx, lm in enumerate(landmarks.landmark):
            csv_writer.writerow([
                frame_idx, kind, idx, lm.x, lm.y, lm.z,
                getattr(lm, 'visibility', '')  # pose has visibility; others don't
            ])

    if results.pose_landmarks:
        write_landmarks(results.pose_landmarks, "pose")
    if results.face_landmarks:
        write_landmarks(results.face_landmarks, "face")
    if results.left_hand_landmarks:
        write_landmarks(results.left_hand_landmarks, "left_hand")
    if results.right_hand_landmarks:
        write_landmarks(results.right_hand_landmarks, "right_hand")

    # --- Save annotated frame ---
    out.write(frame)
    frame_idx += 1

# === Cleanup ===
cap.release()
out.release()
csv_file.close()
holistic.close()
print("✅ Done! Video and CSV saved.")
