import numpy as np
import cv2
from collections import deque


def findMouvement(frame: np.ndarray, frame_buffer:deque, *, THRESHOLD_VALUE=7, MIN_AREA=25): 
    """
    Find the mouvement between the current frame and the previous one
    :param frame: the current frame
    :param frame_buffer: the previous frame
    :return: the frame with the mouvement
    """
    # Convert the frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply a GaussianBlur
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If the buffer is empty, we add the current frame
    if len(frame_buffer) == 0:
        frame_buffer.append(gray)
        return frame

    movement_mask = np.zeros_like(gray)

    # Compare with all frames in buffer
    for prev_frame in frame_buffer:
        diff = cv2.absdiff(gray, prev_frame)
        _, thresh = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        movement_mask = cv2.max(movement_mask, thresh)
    
    # Add current frame to buffer
    frame_buffer.append(gray)
    
    # Process movement mask
    kernel = np.ones((3,3), np.uint8)
    movement_mask = cv2.erode(movement_mask, kernel, iterations=1)
    movement_mask = cv2.dilate(movement_mask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(movement_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

    return contours 


def drawMouvement(frame: np.ndarray, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 

    return frame






import numpy as np
import time
import cv2
from collections import deque
from houghCircle import find2MainCircle

def main():

# Parameters
    BUFFER_SIZE = 1 
    MIN_AREA = 25 
    THRESHOLD_VALUE = 7

    cap = cv2.VideoCapture("./Videos/P1_24h_01top.wmv")
    ret, frame = cap.read()

    frame_buffer = deque(maxlen=BUFFER_SIZE)
    frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    circles = find2MainCircle(frame)
    
    while(cap.isOpened()):
        ret, currentFrame = cap.read()

        if not ret:
            break

        objects = findMouvement(currentFrame, frame_buffer, THRESHOLD_VALUE=THRESHOLD_VALUE, MIN_AREA=MIN_AREA)

        # keep only objects that are circles
        keep = []
        for obj in objects:
            x,y,w,h = cv2.boundingRect(obj)
            center = (x + w//2, y + h//2)
            
            # check if the center of the object is in a circle
            for circle in circles:
                if circle is not None:
                    if np.linalg.norm(np.array(center) - np.array(circle[:2])) < circle[2]:
                        keep.append(obj)
                        break

        objects = keep

        currentFrame = drawMouvement(currentFrame, objects)
        
        for circle in circles:
            if circle is not None:
                print(circle)
               # inner circle
                cv2.circle(currentFrame, (circle[0], circle[1]), circle[2] - 50, (255, 0, 0), 2)
                # outer circles
                cv2.circle(currentFrame, (circle[0], circle[1]), circle[2], (0,0, 255), 2)
                # center of the circle
                cv2.circle(currentFrame, (circle[0], circle[1]), 2, (0, 0, 255), 3)

        cv2.imshow('Movement History', currentFrame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
