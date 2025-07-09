#@markdown To better demonstrate the Pose Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv

def draw_landmarks_on_frame(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='mediapipe_models/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
# image = mp.Image.create_from_file("image.jpg")
input_vid_path = "videos\\1-Introduction-SD.mov"
output_vid_path = "output_vids/position_landmarks.mp4"

csv_file = open("pose_landmarks.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "landmark_index", "x", "y", "z", "visibility"])
landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]

cap = cv2.VideoCapture(input_vid_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_idx = 0

out = cv2.VideoWriter(
    output_vid_path,
    cv2.VideoWriter_fourcc(*'mp4v'),  # Use 'XVID' for .avi
    fps,
    (frame_width, frame_height)
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    timestamp_ms = int((frame_idx / fps) * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    # Draw landmarks on the frame
    annotated = draw_landmarks_on_frame(frame, result)

    # # Write the annotated frame to output video
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    out.write(annotated_bgr)

    if result.pose_landmarks:
        for idx, lm in enumerate(result.pose_landmarks[0]):
            csv_writer.writerow([
            frame_idx,
            landmark_names[idx],
            lm.x, lm.y, lm.z, lm.visibility
        ])
        
        
    frame_idx += 1

    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx} frames")



cap.release()
out.release()
csv_file.close()


# # STEP 4: Detect pose landmarks from the input image.
# detection_result = detector.detect(image)

# # STEP 5: Process the detection result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
# visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
# cv2_imshow(visualized_mask)