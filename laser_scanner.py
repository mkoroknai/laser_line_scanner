import cv2
import numpy as np
import sys
import math
import open3d as o3d
import time


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
            
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            
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

        self.rot_axis_offset = np.array([0, 0, 0])

    

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
        
        points = self.proc_points()
        
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)

        
        distances = pc.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist 
        pc, ind = pc.remove_radius_outlier(nb_points=4, radius=radius)
        
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(.2)
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

        radius = 1.5 * avg_dist

        radii = [radius, 2.0 * radius]
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector(radii))

        o3d.visualization.draw_geometries([pc, rec_mesh], mesh_show_back_face=True)

        timestamp = time.time()
        o3d.io.write_point_cloud("pc_" + str(timestamp) + ".pcd", pc)
        o3d.io.write_triangle_mesh("pc_" + str(timestamp) + ".stl", rec_mesh)

        self.close()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.rot_axis_offset = self.get_xyz(np.array([[y, x]]))[0]
            print(self.rot_axis_offset)

    
    def proc_points(self):
        num_frames = len(self.coords3d)

        for i in range(num_frames):
            self.coords3d[i] = self.coords3d[i][self.coords3d[i][:, 1] < (self.rot_axis_offset[1] + 0.2)]
            self.coords3d[i] = self.coords3d[i][self.coords3d[i][:, 1] > (self.rot_axis_offset[1] - 0.2)]
            self.coords3d[i] = self.coords3d[i][self.coords3d[i][:, 0] < (self.rot_axis_offset[0] + 0.0001)]

        # translate points to center? of coordinate system
        print("offset: " + str(self.rot_axis_offset))
        print(self.coords3d[0][0])
        for i in range(num_frames):
            self.coords3d[i] -= self.rot_axis_offset

        delta_angle = 360.0 / num_frames

        points = self.coords3d[num_frames - 1]
        print("num_frames: " + str(num_frames))
        i = num_frames
        while i > 0:
            rot_mat = rotation_matrix([1, 0, 0], math.radians(i * delta_angle))
            if len(self.coords3d[num_frames - i]) > 0:
                points = np.append(points, np.dot(self.coords3d[num_frames - i], rot_mat.T), axis=0)
                #points = np.append(points, self.coords3d[num_frames - i], axis=0)
            #print( i * delta_angle)
            i -= 1
        
        return points


    def close(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()



def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])