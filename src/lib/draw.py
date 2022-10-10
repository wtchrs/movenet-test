import cv2
import numpy as np
from .colors import EDGE_COLORS


def draw(keypoints, image, result_size=None, threshold=0.11):
    origin_shape = image.shape
    if result_size is not None:
        image = cv2.resize(image, result_size)
    for instance in keypoints:
        denormalized_coordinate = get_denormalized(instance, image.shape, origin_shape[1] / origin_shape[0])
        draw_edges(image, denormalized_coordinate, EDGE_COLORS, threshold)
        draw_keypoints(image, denormalized_coordinate, threshold)
    return image


def get_denormalized(keypoints, target_size, origin_aspect_ratio=None):
    height, width, _ = target_size
    if origin_aspect_ratio is None:
        origin_aspect_ratio = width / height
    if origin_aspect_ratio > 1:
        padding = 0.5 - 0.5 / origin_aspect_ratio
        width_mapping, height_mapping = width, height * origin_aspect_ratio
        remove_padding = np.subtract(keypoints, (padding, 0, 0))
    else:
        padding = 0.5 - 0.5 * origin_aspect_ratio
        width_mapping, height_mapping = width / origin_aspect_ratio, height
        remove_padding = np.subtract(keypoints, (0, padding, 0))
    return np.multiply(remove_padding, (height_mapping, width_mapping, 1))


def draw_keypoints(image, denormalized_coord, threshold=0.11):
    for keypoint in denormalized_coord:
        y, x, confidence = keypoint
        if confidence > threshold:
            cv2.circle(img=image, center=(int(x), int(y)), radius=4, color=(255, 0, 0), thickness=-1)
    return image


def draw_edges(image, denormalized_coord, edge_colors: dict, threshold=0.11):
    for edge, color in edge_colors.items():
        p1, p2 = edge
        y1, x1, confidence1 = denormalized_coord[p1]
        y2, x2, confidence2 = denormalized_coord[p2]
        if (confidence1 > threshold) and (confidence2 > threshold):
            cv2.line(img=image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)),
                     color=color, thickness=2, lineType=cv2.LINE_AA)
