from math import dist
from turtle import pos
import cv2
import numpy as np
import sys
import math


class LaserScanner:

    def __init__(self, camFoV, distCamera2Laser, alphaLaser, camId = None):
        """ angles in radians """
        
        self.resolution = 640
        self.aspect_ratio = float(self.resolution) / 480.0

        self.cam_fov = camFoV
        self.cam_fov_x = camFoV / self.aspect_ratio
        self.d_cl = distCamera2Laser
        self.alpha_l = alphaLaser


        self.tan_0_5fov = math.tan(self.cam_fov / 2.0)
        self.np_0_5 = self.resolution / 2.0
        self.cos_alpha_l = math.cos(self.alpha_l)

        self.tan_0_5fov_x = math.tan(self.cam_fov_x / 2.0)
        self.np_0_5_x = (self.resolution / self.aspect_ratio) / 2.0

        self.cam_id = camId
        if camId is not None:
            self.cap = cv2.VideoCapture(camId)
        else:
            self.cap = None
    

    def get_gamma1(self, pixels_from_center):
        return math.atan(self.tan_0_5fov * (pixels_from_center / self.np_0_5))
    
    def get_phi1(self, pixels_from_center):
        return math.atan(self.tan_0_5fov_x * (pixels_from_center / self.np_0_5_x))

    
    def get_distance(self, gamma1):

        return self.cos_alpha_l * self.d_cl * (math.sin((math.pi / 2.0) - gamma1) / math.sin(gamma1 + self.alpha_l))
    
    def get_xy(self, pixel_coord):

        if pixel_coord[1] < 0:
            return -1
        
        pixels_from_center_y = pixel_coord[1] - self.np_0_5
        gamma1 = self.get_gamma1(pixels_from_center_y)

        pixels_from_center_x = pixel_coord[0] - self.np_0_5_x
        phi1 = self.get_phi1(pixels_from_center_x)

        y = self.get_distance(gamma1)
        x = y * math.tan(phi1)
        
        return x, y
    
    def get_laser_line(self, img, threshold_min, threshold_max):

        green = img[:,:,1]
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ret, frame_thresholded = cv2.threshold(grayscale, threshold_min, threshold_max, cv2.THRESH_BINARY)
        ret, frame_thresholded = cv2.threshold(green, threshold_min, threshold_max, cv2.THRESH_BINARY)
        #np.set_printoptions(threshold=sys.maxsize)
        #print(frame_thresholded)

        non0 = np.transpose(np.nonzero(np.transpose(frame_thresholded)))
        if non0.shape[0] < 1:
            return np.zeros_like(frame_thresholded), []

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

        return frame_lline, lline_data
    
    def get_camera_frame(self):
        if self.cam_id:
            return self.cap.read()