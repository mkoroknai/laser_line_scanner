import cv2
import math
import numpy as np
import open3d as o3d
from laser_scanner import LaserScanner, rotation_matrix




img = cv2.imread("test_rotation_frame.jpg")

THRESHOLD_MIN = 254
THRESHOLD_MAX = 255

CAM_FOV = 64.8 # deg
D_CL = 0.14 # distance between camera and laser position, in m
ALPHA_LASER = 17 # laser tilted by this much, degrees


laser_scanner = LaserScanner(camFoV=math.radians(CAM_FOV), distCamera2Laser=D_CL,
                             alphaLaser=math.radians(ALPHA_LASER))

window_name = "Scan"
window = cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

laser_line, ll_data = laser_scanner.get_laser_line(img, THRESHOLD_MIN, THRESHOLD_MAX)

#coords = []
#for i in range(len(ll_data)):
#    x, y, z = laser_scanner.get_xyz(np.array(ll_data[i]))
#    if y > 0 and y < 0.5:
#        
#        #print([ll_data[i], x, y, z])
#        coords += [[x, y, -z]]

coords = laser_scanner.get_xyz(np.array(ll_data))

# keeping only close coordinates
coords = coords[coords[:, 1] < 0.5]

i = 0

coords[:, 0] -= coords[0, 0]
coords[:, 1] -= coords[0, 1]
coords[:, 2] -= coords[0, 2]

opc = np.copy(coords)

for i in range(180):
    rot_mat = rotation_matrix([1, 0, 0], math.radians(i * 2.0))
    #for j in range(len(opc)):
    #    rotated_point = np.dot(rot_mat, opc[j])
    #    coords = np.append(coords, [rotated_point], axis=0)

    coords = np.append(coords, np.dot(opc, rot_mat.T), axis=0)


while True:

    cv2.imshow(window_name, laser_line)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()


pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(coords)

#pc, ind = pc.remove_radius_outlier(nb_points=2, radius=0.004)
pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

distances = pc.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist

radii = [radius, 2.0 * radius]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector(radii))
#cf = o3d.geometry.TriangleMesh.create_coordinate_frame(.2)
#o3d.visualization.draw_geometries([pc],
#                                  zoom=.72,
#                                  front=[0.1, -0.5, 0.0],
#                                  lookat=[0.0, 0.05, -0.2],
#                                  up=[0, 1, 0])
o3d.visualization.draw_geometries([rec_mesh])