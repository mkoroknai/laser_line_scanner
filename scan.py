import math
from laser_scanner import LaserScanner


THRESHOLD_MIN = 252
THRESHOLD_MAX = 255

CAM_FOV = 64.8 # deg
D_CL = 0.14 # distance between camera and laser position, in m
ALPHA_LASER = 17 # laser tilted by this much, degrees

laser_scanner = LaserScanner(camFoV=math.radians(CAM_FOV), distCamera2Laser=D_CL,
                             alphaLaser=math.radians(ALPHA_LASER), camId=1)


laser_scanner.run()