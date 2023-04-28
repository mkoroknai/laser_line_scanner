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

CAM_FOV = 40 # deg
D_CL = 0.15 # distance between camera and laser position, in m
ALPHA_LASER = 15 # laser tilted by this much, degrees


#cap = cv2.VideoCapture(CAM_ID)


frame = cv2.imread("test_depth03.jpg")

laser_scanner = LaserScanner(math.radians(CAM_FOV), D_CL, math.radians(ALPHA_LASER))

laser_line, lline_data = laser_scanner.get_laser_line(frame, THRESHOLD_MIN, THRESHOLD_MAX)

coords = []

for i in range(len(lline_data)):
    
    x, y = laser_scanner.get_xy(lline_data[i])

    if y > 0:
        coords += [[x, y]]
    #else:
    #    distances += [None]
    #print(str(lline_data[i]) + " : " + str(distance))


window_name = "Scan"
window = cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

coords = np.array(coords)
print(coords)
plt.plot(coords[:, 0], coords[:, 1])
plt.show()

while True:

    #ret, frame = laser_scanner.get_camera_frame()
    #laser_line, lline_data = laser_scanner.get_laser_line(frame, THRESHOLD_MIN, THRESHOLD_MAX)
  
    # Display the resulting frame
    cv2.imshow('Scan', laser_line)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break
  
# After the loop release the cap object
if laser_scanner.cap is not None:
    laser_scanner.cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
