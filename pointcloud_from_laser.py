import cv2
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import open3d as o3d

from laser_scanner import LaserScanner



CAM_ID = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480
THRESHOLD_MIN = 245
THRESHOLD_MAX = 255

CAM_FOV = 64.8 # deg
D_CL = 0.31 # distance between camera and laser position, in m
ALPHA_LASER = 28.67 # laser tilted by this much, degrees


#cap = cv2.VideoCapture(CAM_ID)


frame = cv2.imread("test_depth05.jpg")

laser_scanner = LaserScanner(math.radians(CAM_FOV), D_CL, math.radians(ALPHA_LASER))

laser_line, lline_data = laser_scanner.get_laser_line(frame, THRESHOLD_MIN, THRESHOLD_MAX)

coords = []

for i in range(len(lline_data)):
    
    x, y, z = laser_scanner.get_xyz(lline_data[i])

    #print(lline_data[i])
    print(y)

    if y > 0:
        coords += [[x, y, -z]]

coords = np.array(coords)
surf = coords.copy()
for i in range(20):
    shifted = coords.copy()
    #print(shifted)
    shifted[:, 2] += i / 800.0
    surf = np.append(surf, shifted, axis=0)


#np.set_printoptions(threshold=sys.maxsize)
#print(surf.shape)
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#ax.scatter(surf[:, 0], surf[:, 1], surf[:, 2])
#plt.show()

pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(surf)
pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#print(np.asarray(pc.normals))

#radii = [0.005, 0.01]
#rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([pc], point_show_normal=True)


window_name = "Scan"
window = cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

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
