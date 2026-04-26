import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading
import time

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True).start()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

print("Real-time detection running. Press Q to quit.")

last_speak_time = 0
COOLDOWN_SECONDS = 1.5  # speaks every 1.5 seconds as long as signal is held

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        label = "No signal detected"
        color = (100, 100, 100)

        if result.hand_landmarks:
            landmarks = []
            for hand in result.hand_landmarks:
                for lm in hand:
                    cv2.circle(frame, (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])), 4, (0, 255, 0), -1)
                    landmarks.extend([lm.x, lm.y, lm.z])
            while len(landmarks) < 126:
                landmarks.append(0.0)
            landmarks = landmarks[:126]

            prediction = model.predict([landmarks])[0]
            confidence = max(model.predict_proba([landmarks])[0])

            if confidence > 0.7:
                label = f"{prediction.upper()} ({int(confidence*100)}%)"
                color = (0, 255, 0)

                now = time.time()
                if now - last_speak_time >= COOLDOWN_SECONDS:
                    speak(prediction.replace("_", " "))
                    last_speak_time = now
            else:
                label = f"Uncertain ({int(confidence*100)}%)"
                color = (0, 165, 255)

        cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.imshow("Military Signal Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()