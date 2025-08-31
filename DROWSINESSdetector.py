print("✅ Script started")
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize mixer for alarm
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_alt.xml')
lefteye = cv2.CascadeClassifier('haarcascade_files/haarcascade_lefteye_2splits.xml')
righteye = cv2.CascadeClassifier('haarcascade_files/haarcascade_righteye_2splits.xml')
mouth = cv2.CascadeClassifier('haarcascade_files/haarcascade_smile.xml')

# Load trained model
model = load_model('models/eye_yawn_model_v1.h5')

# Webcam setup
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
path = os.getcwd()

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame. Please check webcam permissions.")
        exit()

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = lefteye.detectMultiScale(gray)
    right_eye = righteye.detectMultiScale(gray)
    mouths = mouth.detectMultiScale(gray, minNeighbors=20, scaleFactor=1.7, minSize=(50, 50))

    eye_closed = False
    yawn_detected = False

    # Check right eye
    for (x, y, w, h) in right_eye:
        eye = frame[y:y+h, x:x+w]
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(eye, (64, 64)) / 255.0
        eye = eye.reshape(1, 64, 64, 1)
        pred = model.predict(eye)
        if np.argmax(pred) == 0:
            eye_closed = True
        break

    # Check left eye
    for (x, y, w, h) in left_eye:
        eye = frame[y:y+h, x:x+w]
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(eye, (64, 64)) / 255.0
        eye = eye.reshape(1, 64, 64, 1)
        pred = model.predict(eye)
        if np.argmax(pred) == 0:
            eye_closed = True
        break

    # Check mouth
    for (x, y, w, h) in mouths:
        mouth_region = frame[y:y+h, x:x+w]
        mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        mouth_region = cv2.resize(mouth_region, (64, 64)) / 255.0
        mouth_region = mouth_region.reshape(1, 64, 64, 1)
        pred = model.predict(mouth_region)
        if np.argmax(pred) == 2:  # Yawn
            yawn_detected = True
        break

    # Scoring
    if eye_closed or yawn_detected:
        score += 1
        status = "Drowsy/Yawning"
    else:
        score -= 1
        status = "Alert"

    score = max(score, 0)

    # Draw status & score
    cv2.putText(frame, status, (10, height - 20), font, 1, (255, 255, 255), 1)
    cv2.putText(frame, f'Score: {score}', (150, height - 20), font, 1, (255, 255, 255), 1)

    # Trigger alarm
    if score > 15:
        cv2.imwrite(os.path.join(path, 'drowsy.jpg'), frame)
        try:
            sound.play()
        except:
            pass

        if thicc < 16:
            thicc += 2
        else:
            thicc = max(thicc - 2, 2)

        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('Drowsiness Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()