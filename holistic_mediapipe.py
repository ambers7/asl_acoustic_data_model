import cv2
import csv
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

input_path = "videos/1-Introduction-SD.mov"
output_path = "output_vids/holistic_output.mp4"
csv_path = "mediapipe_csvs/holistic_landmarks.csv"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'type', 'landmark_index', 'x', 'y', 'z', 'visibility'])

with mp_holistic.Holistic(static_image_mode=False,
                          model_complexity=2,
                          enable_segmentation=False,
                          refine_face_landmarks=False) as holistic:
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Draw landmarks
        annotated = frame.copy()
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Log to CSV
        def log_landmarks(landmarks, name):
            if landmarks:
                for i, lm in enumerate(landmarks.landmark):
                    csv_writer.writerow([frame_idx, name, i, lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else ""])

        log_landmarks(results.pose_landmarks, "pose")
        log_landmarks(results.left_hand_landmarks, "left_hand")
        log_landmarks(results.right_hand_landmarks, "right_hand")

        out.write(annotated)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}")

cap.release()
out.release()
csv_file.close()
print("✅ Done! Holistic video and CSV saved.")
