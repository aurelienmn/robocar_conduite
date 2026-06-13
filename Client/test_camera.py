# coding: utf-8

import cv2

CAMERA_ID = 0

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("Erreur : caméra non détectée")
    exit(1)

print("Caméra détectée. Appuie sur Q pour quitter.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur : impossible de lire l'image")
        break

    cv2.imshow("Test camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()