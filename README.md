import cv2
import mediapipe as mp
import time
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Initialize MediaPipe hand detector
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Setup Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Frame rate
pTime = 0

while True:
    success, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    lmList = []
    distance = 0

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    if len(lmList) >= 9:
        x1, y1 = lmList[4][1], lmList[4][2]   # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]   # Index finger tip

        # Draw circles and line
        cv2.circle(frame, (x1, y1), 10, (255, 0, 255), -1)
        cv2.circle(frame, (x2, y2), 10, (255, 0, 255), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Distance between fingers
        distance = math.hypot(x2 - x1, y2 - y1)

        # Map distance to volume
        vol = np.interp(distance, [50, 250], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)

        # Draw volume bar
        vol_bar = np.interp(distance, [50, 250], [400, 150])
        vol_percent = np.interp(distance, [50, 250], [0, 100])
        cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 2)
        cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1)
        cv2.putText(frame, f'{int(vol_percent)} %', (40, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Add instructions
    cv2.putText(frame, "Move thumb & index finger to control volume", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (480, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display output
    cv2.imshow("Smart Hand Gesture Volume Controller", frame)
    print("Distance between fingers:", distance)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program exited. Camera released and all windows closed.")
 
