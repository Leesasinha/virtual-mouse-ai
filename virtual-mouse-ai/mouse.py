import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

pyautogui.FAILSAFE = False

# Screen size
screen_w, screen_h = pyautogui.size()

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Smooth variables
prev_x, prev_y = 0, 0
smoothening = 10   # 🔥 HIGH = ULTRA SMOOTH

# Click delay
last_click_time = 0
click_delay = 0.8

# ROI margin
frameR = 100

# FPS
pTime = 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Draw ROI box
    cv2.rectangle(frame, (frameR, frameR), (w-frameR, h-frameR), (255,0,255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            lm = handLms.landmark

            # Index finger
            x1 = int(lm[8].x * w)
            y1 = int(lm[8].y * h)

            # Middle finger
            x2 = int(lm[12].x * w)
            y2 = int(lm[12].y * h)

            # Convert to screen coords (ROI mapping)
            screen_x = np.interp(x1, (frameR, w-frameR), (0, screen_w))
            screen_y = np.interp(y1, (frameR, h-frameR), (0, screen_h))

            # Ultra smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            # Dead zone (anti shake)
            if abs(curr_x - prev_x) > 2 or abs(curr_y - prev_y) > 2:
                pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            # Distance for click
            distance = np.hypot(x2 - x1, y2 - y1)

            current_time = time.time()

            # Stable left click
            if distance < 30 and current_time - last_click_time > click_delay:
                pyautogui.click()
                last_click_time = current_time

                cv2.putText(frame, "CLICK", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    # FPS display
    cTime = time.time()
    fps = int(1 / (cTime - pTime + 0.0001))
    pTime = cTime

    cv2.putText(frame, f"FPS: {fps}", (500, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Ultra Smooth Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()