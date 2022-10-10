import cv2
import tensorflow as tf
import tensorflow_hub as hub
import sys
import time
from lib.draw import draw


def main():
    model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    movenet = model.signatures['serving_default']

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.stderr.write('ERROR: Camera is not opened.')
        sys.exit()

    while True:
        start = time.time_ns()

        ret, frame = cap.read()
        if not ret:
            break

        image_resize = tf.image.resize_with_pad(frame, 256, 256)

        image_input = tf.expand_dims(tf.cast(image_resize, dtype=tf.int32), axis=0)
        output = movenet(image_input)
        keypoints = output['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

        height, width, _ = frame.shape

        keypoints_image = draw(keypoints, frame.copy(), (int(width / height * 512), 512), threshold=0.3)
        print(keypoints_image.shape)

        cv2.imshow('Camera Test', keypoints_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time_taken_in_ms = (time.time_ns() - start) / 1_000_000
        fps = 1_000 / time_taken_in_ms
        print(f'Inference Time: {time_taken_in_ms: >6f} ms, FPS: {fps: >6f}')

    cap.release()


if __name__ == '__main__':
    main()
