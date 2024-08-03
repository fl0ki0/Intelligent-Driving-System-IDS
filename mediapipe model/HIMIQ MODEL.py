import cv2 as cv
import numpy as np
import mediapipe as mp
import pygame
from pygame import mixer
import datetime
import csv
import os

# Initialize CSV file
csv_filename = 'attentiveness_data.csv'

def initialize_csv_file():
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Blink Count", "Yawn Count", "Drowsiness Count", "Out of Frame", "Fatigue Status"])

def append_data_to_csv(time, blink_count, yawn_count, drowsiness_count, outof_frame, fatigue_status):
    with open(csv_filename, 'a', newline='') as file:  
        writer = csv.writer(file)
        writer.writerow([time, blink_count, yawn_count, drowsiness_count, outof_frame, fatigue_status])
        
initialize_csv_file()

# Initialize sound
mixer.init()
watch_out = mixer.Sound('watch_out.wav')
eyes_blink = mixer.Sound('wake_up_alarm.mp3')
yawn = mixer.Sound('coffe.wav')
welcome_sound = mixer.Sound('welcome.wav')

# Function to calculate eye openness
def open_len(arr):
    y_arr = [y for _, y in arr]
    min_y = min(y_arr)
    max_y = max(y_arr)
    return max_y - min_y

# Mediapipe initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14]

cap = cv.VideoCapture(0)
welcome_sound.play()

start_time = datetime.datetime.now()

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
) as face_mesh:

    drowsy_frames = 0
    max_left = 0
    max_right = 0
    blink_count = 0
    eye_closed = False
    eye_open_frames = []
    yawn_frames = 0
    yawn_count = 0
    drowsiness_count = 0
    fatigue_status = "Normal"
    left_frames = 0
    right_frames = 0
    down_frames = 0
    threshold_frames = 90
    outof_frame = 0
    screenshot_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        cv.putText(frame, "Hi I'm HIMOQ! your AI Attentive assistant.", (50, img_h - 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                all_landmarks = np.array(
                    [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in face_landmarks.landmark])

                right_eye = all_landmarks[RIGHT_EYE]
                left_eye = all_landmarks[LEFT_EYE]
                upper_lip = all_landmarks[UPPER_LIP]
                lower_lip = all_landmarks[LOWER_LIP]

                cv.polylines(frame, [left_eye], True, (0, 255, 0), 1, cv.LINE_AA)
                cv.polylines(frame, [right_eye], True, (0, 255, 0), 1, cv.LINE_AA)
                cv.polylines(frame, [upper_lip], True, (255, 0, 0), 1, cv.LINE_AA)
                cv.polylines(frame, [lower_lip], True, (255, 0, 0), 1, cv.LINE_AA)

                len_left = open_len(right_eye)
                len_right = open_len(left_eye)
                mouth_openness = open_len(upper_lip) + open_len(lower_lip)

                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv.putText(frame, current_time, (400, 30), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)

                # Use a moving average for eye openness
                eye_open_frames.append((len_left + len_right) / 2)
                if len(eye_open_frames) > 10:  # consider more frames for a smoother average
                    eye_open_frames.pop(0)
                avg_eye_openness = sum(eye_open_frames) / len(eye_open_frames)

                if avg_eye_openness <= (max_left + max_right) / 7:
                    if not eye_closed:
                        blink_count += 1
                        eye_closed = True
                else:
                    eye_closed = False

                if len_left > max_left:
                    max_left = len_left
                if len_right > max_right:
                    max_right = len_right

                # Enhanced drowsiness detection logic
                if (len_left <= max_left / 2 and len_right <= max_right / 2):
                    drowsy_frames += 1
                else:
                    drowsy_frames = 0

                if (drowsy_frames > 60):  # Reduce the threshold for quicker detection
                    drowsiness_count += 1
                    drowsy_frames = 0
                    eyes_blink.play()
                    screenshot_filename = f"drowsiness_screenshot_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{screenshot_counter}.jpg"
                    cv.imwrite(screenshot_filename, frame)
                    screenshot_counter += 1

                if mouth_openness > 50:
                    yawn_frames += 1
                    if yawn_frames > 40:
                        yawn_count += 1
                        yawn_frames = 0
                        yawn.play()
                else:
                    yawn_frames = 0

                if drowsiness_count < 5 and yawn_count < 5:
                    fatigue_status = "Normal"
                elif 5 <= drowsiness_count <= 10 or 5 <= yawn_count <= 10:
                    fatigue_status = "Tired"
                else:
                    fatigue_status = "Very Tired"

                face_2d = []
                face_3d = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        if idx == 1:
                            nose_2d = (x, y)
                            nose_3d = (x, y, lm.z * 3000)
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
                rmat, jac = cv.Rodrigues(rotation_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y >= -10:
                    left_frames = 0
                if y <= 10:
                    right_frames = 0
                if x >= -10:
                    down_frames = 0

                if y < -10:
                    left_frames += 1
                    text = "Looking Left!!!"
                    if left_frames > threshold_frames:
                        watch_out.play()
                        left_frames = 0
                        outof_frame += 1
                elif y > 10:
                    right_frames += 1
                    text = "Looking Right!!!"
                    if right_frames > threshold_frames:
                        watch_out.play()
                        right_frames = 0
                        outof_frame += 1
                elif x < -10:
                    down_frames += 1
                    text = "Looking Down!!!"
                    if down_frames > threshold_frames:
                        watch_out.play()
                        down_frames = 0
                        outof_frame += 1
                else:
                    text = "Forward!!!"

                cv.putText(frame, text, (40, 300), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.putText(frame, f'Blinks: {blink_count}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(frame, f'Yawns: {yawn_count}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(frame, f'Drowsiness: {drowsiness_count}', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(frame, f'Fatigue: {fatigue_status}', (50, 200), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(frame, f'Out of frame: {outof_frame}', (50, 250), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow('img', frame)
        current_time = datetime.datetime.now()
        if (current_time - start_time).total_seconds() >= 20:
            append_data_to_csv(current_time.strftime("%Y-%m-%d %H:%M:%S"), blink_count, yawn_count, drowsiness_count, outof_frame, fatigue_status)
            start_time = current_time  # Reset start time
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()