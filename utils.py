import math
import numpy as np

from shapely.geometry import Point, Polygon


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


def is_point_inside_polygon(point: np.ndarray, polygon: np.ndarray):
    p = Point(point[0], point[1])
    poly = Polygon(polygon)

    return poly.intersects(p)


# def batch_build_polygon(x, y, l, w, heading):
#     coords = 0.5 * np.array([[l, w], [-l, w], [-l, -w], [l, -w]])
#     rotation_matrix = np.array([[math.cos(heading), -math.sin(heading)],
#                                 [math.sin(heading),
#                                  math.cos(heading)]])
#     # rotate
#     coords = np.matmul(coords, rotation_matrix.transpose())

#     # shift
#     coords += np.array([x, y])

#     return coords
