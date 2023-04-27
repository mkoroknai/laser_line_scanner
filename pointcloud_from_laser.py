from math import dist
from turtle import pos
import cv2
import numpy as np
import sys
import math

from laser_scanner import LaserScanner






CAM_ID = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480
THRESHOLD_MIN = 250
THRESHOLD_MAX = 255

CAM_FOV = 50 # deg
D_CL = 0.15 # distance between camera and laser position
ALPHA_LASER = 10 # laser tilted by this much


def get_laser_line(img):
    
    green = img[:,:,1]
    ret, frame_thresholded = cv2.threshold(green, THRESHOLD_MIN, THRESHOLD_MAX, cv2.THRESH_BINARY)
    #np.set_printoptions(threshold=sys.maxsize)
    #print(frame_thresholded)

    non0 = np.transpose(np.nonzero(np.transpose(frame_thresholded)))


    # find laser line by averaging non-zero pixels
    j = 0
    lline_data = []
    for i in range(non0[len(non0) - 1][0] + 1):
        pos_y = 0
        #print(i)
        n_of_pixels_in_col = 0
        while i == non0[j][0]:
            pos_y += non0[j][1]
            #print("j = " + str(j))
            j += 1
            n_of_pixels_in_col += 1
        
            if j == len(non0):
                break
        
        if n_of_pixels_in_col > 0:
            pos_y = pos_y / float(n_of_pixels_in_col)
        else:
            pos_y = -1

        lline_data += [[i, pos_y]]

    frame_lline = np.zeros_like(frame_thresholded)

    for i in range(len(lline_data)):
        y = lline_data[i][1]
        if y > 0:
            frame_lline[round(y), lline_data[i][0]] = 255
        #else:
        #    frame_lline[round(y), lline_data[i][0]] = 0

    return frame_lline




#cap = cv2.VideoCapture(CAM_ID)

window = cv2.namedWindow("Scan", cv2.WINDOW_AUTOSIZE)

frame = cv2.imread("test_depth.png")

laser_scanner = LaserScanner(0, math.radians(40), .15, math.radians(15))

laser_line = laser_scanner.get_laser_line(frame, 250, 255)

while True:

    # Capture the video frame
    # by frame
    #ret, frame = cap.read()

    #print(frame.shape)
  
    # Display the resulting frame
    cv2.imshow('Scan', laser_line)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
#cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
