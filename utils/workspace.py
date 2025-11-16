import numpy as np

VOXEL_RESOLUTION = 64
MAX_POINT_NUM = 1024
WS_X_MIN, WS_Y_MIN,WS_Z_MIN = 0.3, -0.25, -0.02
WS_SIZE = 0.6
WORKSPACE = np.array([[WS_X_MIN, WS_X_MIN + WS_SIZE],
                        [WS_Y_MIN, WS_Y_MIN + WS_SIZE],
                        [WS_Z_MIN, WS_Z_MIN + WS_SIZE]])

VOXEL_SIZE = WS_SIZE/VOXEL_RESOLUTION+1e-4