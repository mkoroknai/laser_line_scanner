from math import dist
from turtle import pos
import cv2
import numpy as np
import sys
import math
import matplotlib.pyplot as plt

from laser_scanner import LaserScanner


CAM_ID = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480
THRESHOLD_MIN = 250
THRESHOLD_MAX = 255

CAM_FOV = 80 # deg
D_CL = 0.2 # distance between camera and laser position
ALPHA_LASER = 10 # laser tilted by this much


#cap = cv2.VideoCapture(CAM_ID)


frame = cv2.imread("test_depth03.jpg")

laser_scanner = LaserScanner(0, math.radians(CAM_FOV), D_CL, math.radians(ALPHA_LASER))

laser_line, lline_data = laser_scanner.get_laser_line(frame, THRESHOLD_MIN, THRESHOLD_MAX)

distances = []

for i in range(len(lline_data)):
    
    distance = laser_scanner.get_distance(lline_data[i][1])

    if distance > 0:
        distances += [distance]
    #else:
    #    distances += [None]
    #print(str(lline_data[i]) + " : " + str(distance))


window_name = "Scan"
window = cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

plt.plot(distances)
plt.show()

while True:

    # Capture the video frame by frame
    #ret, frame = cap.read()

    #print(frame.shape)
  
    # Display the resulting frame
    cv2.imshow('Scan', laser_line)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break
  
# After the loop release the cap object
#cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
