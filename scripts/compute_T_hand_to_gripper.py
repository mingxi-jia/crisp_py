import numpy as np
from scipy.spatial.transform import Rotation as R


# Esther os: [+0.522 -0.013 +0.439]  euler (XYZ°): [ +43.84  +51.40   -0.51]
# [+0.489 +0.000 +0.407]  euler (XYZ°): [+162.08  -72.58 -157.71]
# pos: [+0.497 -0.011 +0.401]  euler (XYZ°): [+151.87  -74.33 -161.64]
                               
hand_euler = np.array([162.08, -72.58, -157.71])
gripper_euler = np.array([180, 0, 0])
hand_xyz = np.array([0.522, -0.013, 0.439])
gripper_xyz = np.array([0.489, 0, 0.407])
R_WH = R.from_euler('XYZ', hand_euler, degrees=True).as_matrix()
R_WG = R.from_euler('XYZ', gripper_euler, degrees=True).as_matrix()


T_WH = np.eye(4)
T_WG = np.eye(4)
T_WH[:3,:3] = R_WH
T_WG[:3,:3] = R_WG
T_WH[:3,3] = hand_xyz
T_WG[:3,3] = gripper_xyz

# Rotation of gripper expressed in hand frame
T_HG = np.linalg.inv(T_WH) @ T_WG

print("T_HG = np.array([")
for row in T_HG:
    print(f"    {list(row)},")
print("])")

