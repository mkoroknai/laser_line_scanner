from calendar import c
import cv2
import numpy as np
import sys
import math


class LaserScanner:

    def __init__(self, camFoV, distCamera2Laser, alphaLaser, camId = None):
        """ angles in radians 
            the scanner assumes a horizontal layout video
            and a vertical laser line
        """

        self.resolution_w = 640.0
        self.resolution_h = 480.0

        self.threshold_min = 250
        self.threshold_max = 255
        
        self.cam_id = camId
        self.window_name = "Laser Scanner"
        self.window_proc_name = self.window_name + " processed"
        if camId is not None:
            self.cap = cv2.VideoCapture(camId)
            self.window = cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.window = cv2.namedWindow(self.window_proc_name, cv2.WINDOW_AUTOSIZE)
            self.resolution_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.resolution_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.frame = None
            
        else:
            self.cap = None
            self.window = None
        
        self.aspect_ratio = float(self.resolution_w) / float(self.resolution_h)

        self.cam_fov = camFoV
        self.cam_fov_x = 2 * math.atan((math.tan(self.cam_fov / 2.0) * self.resolution_h) / self.resolution_w)
        self.d_cl = distCamera2Laser
        self.alpha_l = alphaLaser


        self.tan_0_5fov = math.tan(self.cam_fov / 2.0)
        self.np_0_5 = self.resolution_w / 2.0
        self.cos_alpha_l = math.cos(self.alpha_l)
        #print(self.cos_alpha_l)

        self.tan_0_5fov_x = math.tan(self.cam_fov_x / 2.0)
        self.np_0_5_x = (self.resolution_w / self.aspect_ratio) / 2.0

        self.is_processing = False

    

    def get_gamma1(self, pixels_from_center):
        #print(pixels_from_center/self.np_0_5)
        return np.arctan(self.tan_0_5fov * (pixels_from_center / self.np_0_5))
    
    def get_phi1(self, pixels_from_center):
        return np.arctan(self.tan_0_5fov_x * (pixels_from_center / self.np_0_5_x))

    
    def get_distance(self, gamma1):

        return self.cos_alpha_l * self.d_cl * (np.sin((np.pi / 2.0) - gamma1) / np.sin(gamma1 + self.alpha_l))
    
    def get_xyz(self, pixel_coord):

        if pixel_coord.shape[0] == 0:
            return np.array([])

        coords3D = np.empty((len(pixel_coord), 3))

        pixels_from_center_y = pixel_coord[:, 1] - self.np_0_5
        gamma1 = self.get_gamma1(pixels_from_center_y)
        #print(math.degrees(gamma1))

        pixels_from_center_x = pixel_coord[:, 0] - self.np_0_5_x
        phi1 = self.get_phi1(pixels_from_center_x)

        coords3D[:, 1] = self.get_distance(gamma1)
        coords3D[:, 0] = coords3D[:, 1] * np.tan(phi1)
        coords3D[:, 2] = coords3D[:, 1] * np.tan(gamma1)

        return coords3D
    
    def get_laser_line(self, img, threshold_min, threshold_max):

        green = img[:,:,1]
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ret, frame_thresholded = cv2.threshold(grayscale, threshold_min, threshold_max, cv2.THRESH_BINARY)
        ret, frame_thresholded = cv2.threshold(green, threshold_min, threshold_max, cv2.THRESH_BINARY)
        #np.set_printoptions(threshold=sys.maxsize)
        #print(frame_thresholded)

        non0 = np.transpose(np.nonzero(frame_thresholded))
        if non0.shape[0] < 1:
            return np.zeros_like(frame_thresholded), np.array([])

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

                lline_data += [[i, pos_y]]

        frame_lline = np.zeros_like(frame_thresholded)

        for i in range(len(lline_data)):
            y = lline_data[i][1]
            if y > 0:
                frame_lline[lline_data[i][0], round(y)] = 255
            #else:
            #    frame_lline[round(y), lline_data[i][0]] = 0
        return frame_lline, np.array(lline_data)
    
    def get_camera_frame(self):
        if self.cap.isOpened():
            return self.cap.read()
        else:
            return None, None
    
    def display_camera_frame(self):
        ret, frame = self.get_camera_frame()
        if frame is not None:
            cv2.imshow(self.window_name, frame)
            return ret, frame
        else:
            return None, None
    
    def process_and_display_frame(self, threshold_min, threshold_max):
        ret, frame = self.get_camera_frame()
        if frame is not None:
            laser_line, ll_data = self.get_laser_line(frame, threshold_min, threshold_max)
            cv2.imshow(self.window_name, frame)
            cv2.imshow(self.window_proc_name, laser_line)
            return ret, frame, laser_line, ll_data
        else:
            return None, None, None, None


    def run(self):

        self.coords3d = []

        while True:

            if self.is_processing:
                ret, frame, laser_line, ll_data = self.process_and_display_frame(self.threshold_min, self.threshold_max)
                self.coords3d += [self.get_xyz(ll_data)]
            else:
                self.display_camera_frame()
            
            key = cv2.waitKey(20)

            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord(' '):
                if self.is_processing:
                    break
                else:
                    self.is_processing = True
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    

    def close(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()