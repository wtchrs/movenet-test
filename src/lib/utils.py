import os
import cv2
import numpy as np
from .inference import split_keypoints_bboxes, forwarding_movenet

THRESHOLD = 0.1


def extractFrame(video_lists, save_dir):
    ffmpeg_command = 'ffmpeg -ss 00:00:0 -i {0} -r 10 -f image2 {1}/{2}-%d.jpg'
    for i, video in enumerate(video_lists):
        os.system(ffmpeg_command.format(video, save_dir, i))


def keypointsDataFromImage(movenet, image):
    keypoints, bboxes = split_keypoints_bboxes(forwarding_movenet(movenet, image))
    # keypoints = removePersonWithUnderThreshold(keypoints)
    keypointsList = np.zeros((0, 17, 3))
    bboxesList = np.zeros((0, 5))
    for keypoint, bbox in zip(keypoints, bboxes):
        # print(bbox[4])
        if bbox[4] > THRESHOLD:
            keypointsList = np.append(keypointsList, keypoint.reshape(-1, 17, 3), axis=0)
            bboxesList = np.append(bboxesList, bbox.reshape(-1, 5))
    return keypointsList, bboxesList


def removePersonWithUnderThreshold(coordKeypoints):
    keypoints = np.zeros((0, 17, 3))
    for i in range(coordKeypoints.shape[0]):
        for j in range(coordKeypoints.shape[1]):
            if coordKeypoints[i, j, 2] < THRESHOLD:
                break
            keypoints = np.append(keypoints, coordKeypoints[i, ...].reshape(-1, 17, 3), axis=0)
    return keypoints


def keypointsDataFromImageFiles(movenet, image_lists):
    i = 0
    keypointsData = np.zeros((0, 17, 3))
    for image_path in image_lists:
        i = i + 1
        print(i, image_path)
        image = cv2.imread(image_path)
        keypoints, _ = keypointsDataFromImage(movenet, image)
        keypointsData = np.append(keypointsData, keypoints, axis=0)
        # if i % 10 == 0:
        #     print(keypointsData)
    return keypointsData
