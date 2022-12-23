import itertools

import cv2
import numpy as np
from .colors import EDGE_COLORS


def draw(image, keypoints_out=None, bbox_out=None, result_size=None, threshold=0.11):
    origin_shape = image.shape
    if bbox_out is None:
        bbox_out = []
    if keypoints_out is None:
        keypoints_out = []
    if result_size is not None:
        image = cv2.resize(image, result_size)
    for instance, bbox in itertools.zip_longest(keypoints_out, bbox_out):
        if bbox is not None:
            if bbox[4] < threshold:
                continue
            denormalized_bbox = get_denormalized_bbox(bbox, image.shape, origin_shape[1] / origin_shape[0])
            draw_bbox(image, denormalized_bbox, threshold=threshold)
        if instance is not None:
            denormalized_coordinate = get_denormalized(instance, image.shape, origin_shape[1] / origin_shape[0])
            draw_edges(image, denormalized_coordinate, EDGE_COLORS, threshold)
            draw_keypoints(image, denormalized_coordinate, threshold=threshold)
    return image


def draw_with_denormalized(image, keypoints_out=None, bbox_out=None, result_size=None, threshold=0.11):
    if bbox_out is None:
        bbox_out = []
    if keypoints_out is None:
        keypoints_out = []
    if result_size is not None:
        image = cv2.resize(image, result_size)
    for instance, bbox in itertools.zip_longest(keypoints_out, bbox_out):
        if bbox is not None:
            if bbox[4] < threshold:
                continue
            draw_bbox(image, bbox, threshold=threshold)
        if instance is not None:
            draw_edges(image, instance, EDGE_COLORS, threshold)
            draw_keypoints(image, instance, threshold=threshold)
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


def draw_keypoints(image, denormalized_coord, color=(255, 0, 0), threshold=0.11):
    for keypoint in denormalized_coord:
        y, x, confidence = keypoint
        if confidence > threshold:
            cv2.circle(image, (int(x), int(y)), radius=4, color=color, thickness=-1)
    return image


def draw_edges(image, denormalized_coord, edge_colors: dict, threshold=0.11):
    for edge, color in edge_colors.items():
        p1, p2 = edge
        y1, x1, confidence1 = denormalized_coord[p1]
        y2, x2, confidence2 = denormalized_coord[p2]
        if (confidence1 > threshold) and (confidence2 > threshold):
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2, lineType=cv2.LINE_AA)


def get_denormalized_bbox(coord, target_size, origin_aspect_ratio=None):
    height, width, _ = target_size
    if origin_aspect_ratio is None:
        origin_aspect_ratio = width / height
    if origin_aspect_ratio > 1:
        padding = 0.5 - 0.5 / origin_aspect_ratio
        width_mapping, height_mapping = width, height * origin_aspect_ratio
        remove_padding = np.subtract(coord, (padding, 0, padding, 0, 0))
    else:
        padding = 0.5 - 0.5 * origin_aspect_ratio
        width_mapping, height_mapping = width / origin_aspect_ratio, height
        remove_padding = np.subtract(coord, (0, padding, 0, padding, 0))
    return np.multiply(remove_padding, (height_mapping, width_mapping, height_mapping, width_mapping, 1))


def draw_bbox(image, denormalized_bbox, box_color=(0, 0, 255), threshold=0.11):
    y1, x1, y2, x2, confidence = denormalized_bbox
    if confidence > threshold:
        points = [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for edge in edges:
            cv2.line(image, points[edge[0]], points[edge[1]], color=box_color, thickness=2, lineType=cv2.LINE_AA)
