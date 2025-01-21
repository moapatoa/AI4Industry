import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('SOLVAY/P1_24h_02mid.wmv')

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=6, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background subtractor to get the foreground mask
    fgmask = fgbg.apply(frame)

    # Apply some morphological operations to remove noise and fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

    # Erosion + Dialation
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Trouver les contours dans le masque
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pour dessiner
    for contour in contours:
        if cv2.contourArea(contour) > 65:  # Lowered the threshold => detecte moins facilement
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectangles Verts

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == 27: # Echap pour quitter
        break

cap.release()
cv2.destroyAllWindows()