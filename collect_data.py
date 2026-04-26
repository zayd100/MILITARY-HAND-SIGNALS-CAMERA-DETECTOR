import cv2
import mediapipe as mp
import csv
import os

SIGNALS = ["stop", "move_out", "enemy", "down", "come_here"]

print("Available signals:")
for i, s in enumerate(SIGNALS):
    print(f"  {i}: {s}")

signal_name = SIGNALS[int(input("Enter signal number to record: "))]
os.makedirs("data", exist_ok=True)
output_file = f"data/{signal_name}.csv"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

collected = 0
TARGET = 200

print(f"\nRecording '{signal_name}'. Press SPACE to capture. Need {TARGET} samples.")

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        for hand in result.hand_landmarks:
            for lm in hand:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        cv2.putText(frame, f"Signal: {signal_name} | Samples: {collected}/{TARGET}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE = capture | Q = quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and result.hand_landmarks:
            landmarks = []
            for hand in result.hand_landmarks:
                for lm in hand:
                    landmarks.extend([lm.x, lm.y, lm.z])
            while len(landmarks) < 126:
                landmarks.append(0.0)
            landmarks = landmarks[:126]
            with open(output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([signal_name] + landmarks)
            collected += 1
            print(f"  Captured {collected}/{TARGET}")
            if collected >= TARGET:
                print("Done!")
                break

    cap.release()
    cv2.destroyAllWindows()