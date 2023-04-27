from math import dist
from turtle import pos
import cv2
import numpy as np
import sys
import math


class LaserScanner:

    def __init__(self, camId, camFoV, distCamera2Laser, alphaLaser):
        """ please enter angles in radians """
        
        self.cam_id = camId
        self.cam_fov = camFoV
        self.d_cl = distCamera2Laser
        self.alpha_l = alphaLaser

        self.horizontal_resolution = 640

        self.tan_0_5fov = math.tan(self.cam_fov / 2.0)
        self.np_0_5 = self.horizontal_resolution / 2.0
        self.cos_alpha_l = math.cos(self.alpha_l)
    

    def get_gamma1(self, pixels_from_center):
        return math.atan(self.tan_0_5fov * (pixels_from_center / self.np_0_5))
    
    def get_distance(self, pixels_from_center):

        gamma1 = self.get_gamma1(pixels_from_center)

        return self.cos_alpha_l * self.d_cl * (math.sin((math.pi / 2.0) - gamma1) / math.sin(gamma1 + self.alpha_l))
    
    def get_laser_line(self, img, threshold_min, threshold_max):

        green = img[:,:,1]
        ret, frame_thresholded = cv2.threshold(green, threshold_min, threshold_max, cv2.THRESH_BINARY)
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