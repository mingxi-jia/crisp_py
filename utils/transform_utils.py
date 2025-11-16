from typing import List, Union
from scipy.spatial.transform import Rotation as R
import numpy as np

def create_transform_matrix(translation: Union[List[float], np.ndarray], quaternion: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Creates a 4x4 SE(3) transformation matrix from translation and quaternion.

    Args:
        translation (Union[List[float], np.ndarray]): A (3,) translation vector [x, y, z].
        quaternion (Union[List[float], np.ndarray]): A (4,) quaternion [qx, qy, qz, qw].

    Returns:
        np.ndarray: The (4, 4) transformation matrix.
    """
    if len(translation) != 3 or len(quaternion) != 4:
        raise ValueError("Translation must be of length 3 and quaternion of length 4.")
    
    transform = np.eye(4)
    transform[:3, :3] = R.from_quat(quaternion).as_matrix()
    transform[:3, 3] = translation
    
    return transform

def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Applies an SE(3) transformation to a set of points.

    Args:
        points (np.ndarray): An (N, 3) or (N, 6) array of points.
        transform (np.ndarray): A (4, 4) transformation matrix.

    Returns:
        np.ndarray: The transformed points, maintaining original shape.
    """
    if points.shape[0] == 0:
        return points
        
    points_copy = points.copy()
    xyz = points_copy[:, :3]
    
    points_homogeneous = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    transformed_points = (transform @ points_homogeneous.T).T
    
    points_copy[:, :3] = transformed_points[:, :3]
    return points_copy