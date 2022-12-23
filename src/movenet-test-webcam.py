import os
import sys
import cv2
import time
from lib.draw import draw
from lib.inference import load_movenet, forwarding_movenet, split_keypoints_bboxes

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main():
    movenet = load_movenet()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.stderr.write('ERROR: Camera is not opened.')
        sys.exit()

    while True:
        start = time.time_ns()

        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        keypoints, bboxes = split_keypoints_bboxes(forwarding_movenet(movenet, frame))
        keypoints_image = draw(frame.copy(), keypoints, bboxes, result_size=(int(width / height * 512), 512),
                               threshold=0.3)

        cv2.imshow('Camera Test', keypoints_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time_taken_in_ms = (time.time_ns() - start) / 1_000_000
        fps = 1_000 / time_taken_in_ms
        print(f'\rInference Time: {time_taken_in_ms: .2f} ms, FPS: {fps: .2f}', end='')

    cap.release()


if __name__ == '__main__':
    main()
