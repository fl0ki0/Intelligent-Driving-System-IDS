import cv2 as cv
import numpy as np
import mediapipe as mp
import datetime
import time
import csv
import os
import pandas as pd
import pickle
import glob
import scipy.io
from google.cloud import firestore
import google.auth
import threading
from gpiozero import Buzzer
import serial
# Configure serial port settings
ser = serial.Serial(
    port='/dev/ttyAMA0',  # Replace with the correct port if needed (e.g., /dev/ttyUSB0)
    baudrate=115200,
    timeout=1  # Set a timeout to avoid waiting indefinitely for responses
)
csv_filename = 'attentiveness_data.csv'
buzzer_pin = 4  # Replace with your actual buzzer pin if different
buzzer = Buzzer(buzzer_pin)

def check_for_termination_flag():
  # Check if the flag file exists
  if os.path.exists("terminate_flag.txt"):
    # Open the file and read its content
    with open("terminate_flag.txt", "r") as f:
      flag_content = f.read().strip()
    # Check if the flag content is "q"
    if flag_content == "q":
      print("Termination signal received from PIR script. Exiting...")
      update_sessions(blink_count, yawn_count, drowsiness_count, fatigue_status, outof_frame)
      with open("terminate_flag.txt", "w") as f:
       f.write("")
      time.sleep(3)
      return True  # Indicate termination
  return False  # No termination signal found

current_time = datetime.datetime.now()
# Firebase project ID
PROJECT_ID = "driver-live-j61jka"
SERVICE_ACCOUNT_KEY_PATH = "/home/pi/Downloads/driver-live-j61jka-firebase-<some auth key>.json"

def get_firestore_client():
    # Initialize Firestore client with service account credentials
    credentials, _ = google.auth.load_credentials_from_file(SERVICE_ACCOUNT_KEY_PATH)
    return firestore.Client(credentials=credentials, project=PROJECT_ID)

def send_to_firestore(alert_type, location, state, collection="Posts"):
    try:
        # Initialize Firestore client
        db = get_firestore_client()

        # Get current timestamp
        current_time = datetime.datetime.now()
        
        # Generate document name based on timestamp
        document_name = current_time.strftime("%Y-%m-%dT%H:%M:%S")

        # Construct document reference
        doc_ref = db.collection(collection).document(document_name)

        # Data to be sent
        data = {
            "alertType": alert_type,
            "location": location,
            "state": state,
            "timestamp": current_time
        }

        # Send data to Firestore
        doc_ref.set(data)
        print(f"Data sent successfully to Firestore in document '{document_name}'.")

    except Exception as e:
        print("An error occurred:", e)

def update_sessions(blink_count, yawn_count, drowsiness_count, fatigue_status, outof_frame):
    try:
        # Initialize Firestore client
        db = get_firestore_client()

        # Get current date for session
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Create document name based on current date
        session_doc_name = current_date

        # Construct document reference for the session
        session_doc_ref = db.collection("Sessions").document(session_doc_name)

        # Get the existing document
        session_doc = session_doc_ref.get()

        if session_doc.exists:
            # Update existing document
            session_data = session_doc.to_dict()
            session_data["blink_count"] += blink_count
            session_data["yawn_count"] += yawn_count
            session_data["drowsiness_count"] += drowsiness_count
            session_data["outof_frame"] += outof_frame
            session_data["fatigue_status"] = fatigue_status  # Optional: You can decide how to handle this
        else:
            # Create new document
            session_data = {
                "timestamp": current_date,
                "blink_count": blink_count,
                "yawn_count": yawn_count,
                "drowsiness_count": drowsiness_count,
                "fatigue_status": fatigue_status,
                "outof_frame": outof_frame
            }

        # Send session data to Firestore
        session_doc_ref.set(session_data)
        print(f"Session data sent successfully to Firestore in document '{session_doc_name}'.")

    except Exception as e:
        print("An error occurred in session counter update:", e)
        
BUZZER_DURATION = 4

def buzz_for_seconds(seconds):
    global buzzer_active
    buzzer_active = True
    buzzer.on()
    time.sleep(seconds)
    buzzer.off()
    buzzer_active = False          
def trigger_firestore(alert_type, location, state):
    threading.Thread(target=send_to_firestore, args=(alert_type, location, state)).start()
        
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
# Model loading
model = pickle.load(open('./testingposemodel.pkl', 'rb'))
cols = []
for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']:
    for dim in ('x', 'y'):
        cols.append(pos + dim)
def emergency_call():
    try:
        # Send Call command
        print("Sending call command...")
        ser.write('ATD+201124542598\r\n'.encode('latin-1'))
        response = ser.read(1024).decode('latin-1')
        print("ATD response:", response.strip())

        # Send turn on GPS command
        print("Sending GPS command...")
        ser.write('AT+GPS=1\r\n'.encode('latin-1'))
        response = ser.read(1024).decode('latin-1')
        print("AT+GPS response:", response.strip())

        # Send get Long,Lat command
        print("Sending location command...")
        ser.write('AT+LOCATION=2\r\n'.encode('latin-1'))
        time.sleep(3)
        response = ser.read(1024).decode('latin-1')
        print("AT+LOCATION response:", response.strip())

        if response and (response[9] != ',' or response[0] == 'ÿ'):
            print('Location data unavailable.')
            ser.write('AT+CMGF=1\r\n'.encode('latin-1'))
            time.sleep(1)
            response = ser.read(1024).decode('latin-1')
            ser.write('AT+CNMI=2,2,0,0,0\r\n'.encode('latin-1'))
            time.sleep(1)
            response = ser.read(1024).decode('latin-1')
            ser.write('AT+CMGS=+201124542598\r\n'.encode('latin-1'))
            response = ser.read(1024).decode('latin-1')
            time.sleep(1)
            ser.write('https://maps.google.com/?q=29.987526,31.327370\r\n'.encode('latin-1') + b"\x1A")
        else:
            # Extract latitude and longitude from the response
            data_parts = response.split(",")
            if len(data_parts) == 2:
                latitude = float(data_parts[0])
                longitude = float(data_parts[1])
                ser.write('AT+CMGF=1\r\n'.encode('latin-1'))
                time.sleep(1)
                response = ser.read(1024).decode('latin-1')
                ser.write('AT+CNMI=2,2,0,0,0\r\n'.encode('latin-1'))
                time.sleep(1)
                response = ser.read(1024).decode('latin-1')
                ser.write('AT+CMGS=+201124542598\r\n'.encode('latin-1'))
                time.sleep(1)
                ser.write(f'https://maps.google.com/?q={latitude},{longitude}\r\n'.encode('latin-1') + b"\x1A")

    except serial.SerialException as e:
        print(f"Serial error occurred: {e}")
        # Log the error or handle it as needed (e.g., retry, continue, etc.)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Handle other exceptions if necessary

def extract_features(img, face_mesh):
    poseNOSE = 1
    poseFOREHEAD = 10
    poseLEFT_EYE = 33
    poseMOUTH_LEFT = 61
    poseCHIN = 199
    poseRIGHT_EYE = 263
    poseMOUTH_RIGHT = 291

    result = face_mesh.process(frame)
    face_features = []
    
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [poseFOREHEAD, poseNOSE, poseMOUTH_LEFT, poseMOUTH_RIGHT, poseCHIN, poseLEFT_EYE, poseRIGHT_EYE]:
                    face_features.append(lm.x)
                    face_features.append(lm.y)

    return face_features

def normalize(poses_df):
    normalized_df = poses_df.copy()
    
    for dim in ['x', 'y']:
        # Centering around the nose 
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] = poses_df[feature] - poses_df['nose_'+dim]
        
        # Scaling
        diff = normalized_df['mouth_right_'+dim] - normalized_df['left_eye_'+dim]
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] = normalized_df[feature] / diff
    
    return normalized_df


def open_len(arr):
    y_arr = [y for _, y in arr]
    min_y = min(y_arr)
    max_y = max(y_arr)
    return max_y - min_y

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14]

cap = cv.VideoCapture(0)

start_time = datetime.datetime.now()

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
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
    up_frames = 0
    threshold_frames = 35
    outof_frame=0
    screenshot_counter = 0
    buzzer_state = False
    longlat = "location"
    i = 0
    lati = 30.033333
    longi = 31.233334
    last_trigger_time = None
    last_drowsy_frame = 0
    buzzer_active = False
    Sess_up = False
    os.environ["HIMOQ_Sess"] = str(Sess_up)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        cv.putText(frame, "Hi I'm HIMOQ! your AI Attentive assistant.", (50, img_h - 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        results = face_mesh.process(rgb_frame)
        # Extract face features
        face_features = extract_features(rgb_frame, face_mesh)
        if len(face_features) > 0:
         # Prepare features dataframe and normalize
            face_features_df = pd.DataFrame([face_features], columns=cols)
            face_features_normalized = normalize(face_features_df)

            # Predict pitch, yaw, roll
            pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized).ravel()
        
            # Draw axes on the image based on predicted head pose
            img_h, img_w, _ = frame.shape
            nose_x = face_features_df['nose_x'].values * img_w
            nose_y = face_features_df['nose_y'].values * img_h
            img = draw_axes(frame, pitch_pred, yaw_pred, roll_pred, nose_x, nose_y)

            # Display text based on head pose
            text = ''
            if pitch_pred > 0.6:
                text = 'Top'
                up_frames += 1
                if up_frames > threshold_frames:
                    buzzer.on()
                    trigger_firestore("Distraction Alert","None",fatigue_status)
                    up_frames = 0  
                    outof_frame +=1
                if yaw_pred > 0.6:
                    text = 'Top Left'
                elif yaw_pred < -0.6:
                    text = 'Top Right'
            elif pitch_pred < -0.3:
                text = 'Bottom'
                down_frames += 1
                if down_frames > threshold_frames:
                    threading.Thread(target=buzz_for_seconds, args=(BUZZER_DURATION,)).start()
                    if (down_frames%55 == 0):
                           outof_frame +=1
                           trigger_firestore("Distraction Alert","None",fatigue_status)
                    if down_frames%160 == 0:
                           emergency_call()
                           trigger_firestore("Driver Emergency","https://www.google.com/maps/search/shorouk+academy+google+maps/@30.1244651,31.5979621,14z?entry=ttu",fatigue_status)
                if yaw_pred > 0.3:
                    text = 'Bottom Left'
                elif yaw_pred < -0.3:
                    text = 'Bottom Right'
            elif yaw_pred > 0.6:
                text = 'Left'
                left_frames += 1
                if left_frames > threshold_frames:
                    buzzer.on()
                    trigger_firestore("Distraction Alert","None",fatigue_status)
                    left_frames = 0  
                    outof_frame +=1
            elif yaw_pred < -0.6:
                text = 'Right'
                right_frames += 1
                if right_frames > threshold_frames:
                    buzzer.on()
                    trigger_firestore("Distraction Alert","None",fatigue_status)
                    right_frames = 0  
                    outof_frame +=1
            else:
                text = 'Forward'
                buzzer.off()
                buzzer_active = False
                down_frames = 0 
                
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

           

                eye_open_frames.append((len_left + len_right) / 2)
                if len(eye_open_frames) > 7:
                    eye_open_frames.pop(0)
                avg_eye_openness = sum(eye_open_frames) / len(eye_open_frames)
                if avg_eye_openness <= (max_left + max_right) / 6 :
                    if not eye_closed:
                        blink_count += 1
                        eye_closed = True
                else:
                    eye_closed = False

                if len_left > max_left:
                    max_left = len_left
                if len_right > max_right:
                    max_right = len_right

                if (len_left <= int(max_left / 2)+1) and (len_right <= int(max_right / 2)+1):
                    drowsy_frames += 1
                else:
                    drowsy_frames = 0

                if (drowsy_frames > 44):
                    threading.Thread(target=buzz_for_seconds, args=(BUZZER_DURATION,)).start()
                    if (drowsy_frames%45 == 0):
                        drowsiness_count += 1
                        trigger_firestore("Drowsiness Alert","None",fatigue_status)
                    if drowsy_frames%160 == 0:
                        drowsy_frames = 0
                        emergency_call()
                        trigger_firestore("Driver Emergency","https://www.google.com/maps/search/shorouk+academy+google+maps/@30.1244651,31.5979621,14z?entry=ttu",fatigue_status)
                if (len_left > int(max_left / 2)+1) and (len_right > int(max_right / 2)+1):
                    drowsy_frames = 0  # Reset drowsy_frames only when not triggered
                    last_drowsy_frame = 0
                    max_left = len_left+1
                    max_right = len_right+1

                if mouth_openness > 50:
                    yawn_frames += 1
                    if yawn_frames > 40:
                        yawn_count += 1
                        yawn_frames = 0
                        trigger_firestore("Yawn","None",fatigue_status)
                else:
                    yawn_frames = 0

                if drowsiness_count < 5 and yawn_count < 5:
                    fatigue_status = "Normal"
                elif 5 <= drowsiness_count <= 10 or 5 <= yawn_count <= 10:
                    fatigue_status = "Tired"
                else:
                    fatigue_status = "Very Tired"
                if os.environ["HIMOQ_Sess"] == True:
                    update_sessions(blink_count, yawn_count, drowsiness_count, fatigue_status, outof_frame)
                    


                cv.putText(frame, text, (40,300), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2)
                cv.putText(frame, f'Slow Blinks: {blink_count}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(frame, f'Yawns: {yawn_count}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(frame, f'Sleep/Drowsiness: {drowsiness_count}', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(frame, f'Fatigue : {fatigue_status}', (50, 200), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(frame, f'distraction : {outof_frame}', (50, 250), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow('img', frame)
        os.environ["HIMOQ_BLINK_COUNT"] = str(blink_count)
        os.environ["HIMOQ_YAWN_COUNT"] = str(yawn_count)
        os.environ["HIMOQ_DROWSINESS_COUNT"] = str(drowsiness_count)
        os.environ["HIMOQ_OUTOF_FRAME"] = str(outof_frame)
        os.environ["HIMOQ_FATIGUE_STATUS"] = str(fatigue_status)
        current_time = datetime.datetime.now()
        if (current_time - start_time).total_seconds() >= 20:
              append_data_to_csv(current_time.strftime("%Y-%m-%d %H:%M:%S"), blink_count, yawn_count, drowsiness_count, outof_frame, fatigue_status)
              start_time = current_time  # Reset start time
        if cv.waitKey(1) & 0xFF == ord('q'):
            update_sessions(blink_count, yawn_count, drowsiness_count, fatigue_status, outof_frame)
            time.sleep(3)
            break
        if check_for_termination_flag():
            break

cap.release()
