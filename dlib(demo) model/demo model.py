from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import time

mixer.init()
mixer.music.load(r"C:\Users\LOVE\Downloads\dlib\alarm.wav")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[12], mouth[16])
    mar = A / C
    return mar


def detect_drowsiness_and_yawn(gray, shape):
    global flag_ear, flag_mar, alarm_start_time, total_drowsy_time, total_yawn_count

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    mouth = shape[mStart:mEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0

    mar = mouth_aspect_ratio(mouth)

    # Display the calculated EAR and MAR
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Change color to red if eyes are closed or yawning
    color = (0, 255, 0)  # Default color is green
    if ear < thresh_ear or mar > thresh_mar:
        flag_ear += 1
        flag_mar += 1
        print(flag_ear, flag_mar)
        if flag_ear >= frame_check or flag_mar >= frame_check:
            # Change color to red when eyes are closed or yawning
            color = (0, 0, 255)
            cv2.putText(frame, "****************Wake up!****************", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "****************Wake up!****************", (10, 325),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mixer.music.play()

            # Set the alarm start time when drowsiness is detected
            if alarm_start_time is None:
                alarm_start_time = time.time()

        # Check if the alarm should be enabled
        if alarm_start_time is not None and time.time() - alarm_start_time > alarm_duration:
            cv2.putText(frame, "****************Alarm Enabled!****************", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Add your code to enable the alarm here

            # Display the time elapsed during drowsinessqqqq
            total_drowsy_time += 1
            cv2.putText(frame, f"Drowsy Time: {total_drowsy_time}s", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    else:
        flag_ear = 0
        flag_mar = 0
        alarm_start_time = None

    # Check for yawning
    if mar > thresh_mar:
        total_yawn_count += 1
        cv2.putText(frame, f"Yawn Count: {total_yawn_count}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Display a warning message for yawning
        cv2.putText(frame, "Warning: Yawning detected! Focus or drink some coffee.", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw contours with the selected color
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, color, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, color, 1)
    cv2.drawContours(frame, [mouth], -1, color, 1)


# Set the duration for the alarm to be enabled (in seconds)
alarm_duration = 8

# Define thresholds for EAR and MAR
thresh_ear = 0.25
thresh_mar = 0.5

# Set the frame check duration
frame_check = 20

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(
    r"C:\Users\LOVE\Downloads\dlib\models-20231206T220641Z-001\models\shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

cap = cv2.VideoCapture(0)
flag_ear = 0
flag_mar = 0
alarm_start_time = None
total_drowsy_time = 0
total_yawn_count = 0

# Set window to full screen
cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        detect_drowsiness_and_yawn(gray, shape)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()