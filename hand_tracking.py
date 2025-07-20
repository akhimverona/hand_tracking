import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    message_parts = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            h, w, c = img.shape
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            if lmList:
                if lmList[8][2] < lmList[6][2]:
                    message_parts.append("I")
                if lmList[12][2] < lmList[10][2]:
                    message_parts.append("Love")
                if lmList[16][2] < lmList[14][2]:
                    message_parts.append("You")

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    if message_parts:
        full_message = "Message: " + " ".join(message_parts)
        cv2.putText(img, full_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

    # FPS Display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 450), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q')
        break

cap.release()
cv2.destroyAllWindows()
