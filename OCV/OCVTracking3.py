import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('Videos/P1_24h_02mid.wmv')

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=6, detectShadows=True)

# List to store the positions of rectangles
rect_positions = []

# Number of frames to keep the trace
trace_length = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background subtractor to get the foreground mask
    fgmask = fgbg.apply(frame)

    # Apply some morphological operations to remove noise and fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Erosion + Dilation
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Trouver les contours dans le masque
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fusionner les contours proches
    merged_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 70:
            merged = False
            for i, existing_contour in enumerate(merged_contours):
                a, t = contour[0][0]
                if cv2.pointPolygonTest(existing_contour, (a / 100, t / 100), True) >= -15:
                    merged_contours[i] = np.vstack((existing_contour, contour))
                    merged = True
                    break
            if not merged:
                merged_contours.append(contour)

    # Pour dessiner
    current_rects = []
    for contour in merged_contours:
        if cv2.contourArea(contour) > 65:  # Lowered the threshold => detecte moins facilement
            x, y, w, h = cv2.boundingRect(contour)
            current_rects.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectangles Verts

    # Add current rectangles to the list of positions
    rect_positions.append(current_rects)

    # Keep only the last 'trace_length' frames
    if len(rect_positions) > trace_length:
        rect_positions.pop(0)

    # Draw the trace of rectangles
    for rects in rect_positions:
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Echap pour quitter
        break

cap.release()
cv2.destroyAllWindows()