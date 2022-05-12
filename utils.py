import math
import numpy as np


def build_polygon(x, y, l, w, heading):
    coords = 0.5 * np.array([[l, w], [-l, w], [-l, -w], [l, -w]])

    rotation_matrix = np.array([[math.cos(heading), -math.sin(heading)],
                                [math.sin(heading),
                                 math.cos(heading)]])
    # rotate
    coords = np.matmul(coords, rotation_matrix.transpose())

    # shift
    coords += np.array([x, y])

    return coords


def batch_build_polygon(inputs: np.ndarray) -> np.ndarray:
    """build rectangular polygon for batch inputs

    Args:
        inputs: [bs, 5] with x, y, length, width, heading
    
    Returns:
        polygon: [bs, 4, 2]
    """
    xy = inputs[:, :2]
    l = inputs[:, 2]
    w = inputs[:, 3]
    heading = inputs[:, 4]

    # shape [n, 4, 2]
    coords = 0.5 * np.column_stack([l, w, -l, w, -l, -w, l, -w]).reshape(
        -1, 4, 2)

    rotate_mat = np.column_stack(
        [np.cos(heading), -np.sin(heading),
         np.sin(heading),
         np.cos(heading)]).reshape(-1, 2, 2)

    # rotate
    coords = np.matmul(coords, rotate_mat.transpose(0, 2, 1))
    # shift
    coords += np.expand_dims(xy, axis=1)

    return coords
