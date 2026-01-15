from collections import deque
import numpy as np
import cv2


# Taille de l'historique pour tracer la trajectoire (plus grand = plus longue trace)
MAX_POINTS = 64
points = deque(maxlen=MAX_POINTS)


canvas = None
paused = False
# Plage HSV pour une couleur (ici: bleu par défaut)
# HSV = (Hue, Saturation, Value)

LOWER_HSV = np.array([105, 80, 20])
UPPER_HSV = np.array([125, 255, 120])


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam access deny")

MAX_POINTS = 64
points = deque(maxlen=MAX_POINTS)


while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Frame non lue, on continue...")
            continue

        frame = cv2.flip(frame, 1)

        # Initialisation du canvas UNE SEULE FOIS
        if canvas is None:
            h, w = frame.shape[:2]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Pré-traitement

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


        # Masque couleur

        mask1 = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        mask1 = cv2.erode(mask1, None, iterations=2)
        mask1 = cv2.dilate(mask1, None, iterations=2)


        # Contours

        contours, _ = cv2.findContours(
            mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        center = None

        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (
                    int(M["m10"] / M["m00"]),
                    int(M["m01"] / M["m00"]),
                )
            else:
                center = (int(x), int(y))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)


        # Stocker & dessiner trajectoire (PERSISTANTE)

        if center is not None:
            points.appendleft(center)

        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            thickness = int(np.sqrt(MAX_POINTS / float(i + 1)) * 2)
            cv2.line(canvas, points[i - 1], points[i], (0, 255, 0), thickness)

    # Overlay canvas + frame
    output = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)

    cv2.imshow("Tracking", output)
    cv2.imshow("Mask", mask1)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas[:] = 0
        points.clear()
    elif key == ord("p"):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
