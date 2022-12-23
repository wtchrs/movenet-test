import cv2
import time
import os
import sys
import argparse
import pygame
from pathlib import Path
from keras.models import load_model
from lib.utils import keypointsDataFromImage
from lib.video import BufferlessVideoCapture
from lib.inference import load_movenet
from lib.draw import draw, get_denormalized


def main():
    root_dir = str(Path(os.path.dirname(os.path.realpath(__file__))) / '..')

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', default=None, help='RTSP url as input')
    args = vars(parser.parse_args())

    movenet = load_movenet()

    model = load_model(root_dir + '/out/model/checkpoint.h5')

    pygame.init()
    pygame.mixer.init()
    sound = pygame.mixer.Sound(root_dir + '/mp3/alarm.mp3')
    sound.set_volume(0.9)

    cam_url = args.get('url')
    if cam_url is None:
        cam_url = 0
    capture = BufferlessVideoCapture(cam_url)

    if not capture.isOpened():
        sys.stderr.write('ERROR: Camera is not opened.')
        sys.exit()

    dumpLabel = 'Dumping: {0:.2f}'
    nodumpLabel = 'Person'

    buzzerTime = time.time()
    while True:
        t = time.time()

        # ret, frame = cap.read()
        frame = capture.read()
        # if not ret:
        #     break

        keypoints, bboxes = keypointsDataFromImage(movenet, frame)
        denormalized_keypoints = get_denormalized(keypoints, frame.shape)
        if keypoints.size != 0:
            pred = model.predict(keypoints, verbose=0)
        else:
            pred = []

        for i, prob in enumerate(pred):
            font_size = 1
            font_thickness = 2
            if prob[0] > 0.5:
                if time.time() - buzzerTime > 1:
                    buzzerTime = time.time()
                    sound.play()
                headPosition = denormalized_keypoints[i, 0, 0:2]
                headPosition = (int(headPosition[1]), int(headPosition[0]))
                labelWithProb = dumpLabel.format(prob[0] * 100)
                (w, h), _ = cv2.getTextSize(labelWithProb, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
                frame = cv2.rectangle(frame, (headPosition[0], headPosition[1] - h),
                                      (headPosition[0] + w, headPosition[1]), (0, 0, 255), -1)
                frame = cv2.putText(frame, labelWithProb, (headPosition[0], headPosition[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)
            else:
                headPosition = denormalized_keypoints[i, 0, 0:2]
                headPosition = (int(headPosition[1]), int(headPosition[0]))
                labelWithProb = nodumpLabel.format(prob[0] * 100)
                (w, h), _ = cv2.getTextSize(labelWithProb, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
                frame = cv2.rectangle(frame, (headPosition[0], headPosition[1] - h),
                                      (headPosition[0] + w, headPosition[1]), (47, 157, 39), -1)
                frame = cv2.putText(frame, labelWithProb, (headPosition[0], headPosition[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)

        frame1 = draw(frame.copy(), keypoints)

        cv2.imshow('Real-time Video', frame)
        cv2.imshow('With skeleton', frame1)

        timeTaken = time.time() - t
        print(f'FPS = {1 / timeTaken}, Time Taken = {timeTaken}', end='\r')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()


if __name__ == '__main__':
    main()
