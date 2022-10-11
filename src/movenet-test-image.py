import cv2
import tensorflow as tf
import tensorflow_hub as hub
import os
from glob import glob
from lib.draw import draw, get_denormalized, draw_keypoints


def main():
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    image_path = current_file_path + '/../test_file/image'
    images = glob(image_path + '/*.jpg')

    model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    movenet = model.signatures['serving_default']

    for img in images:
        frame = cv2.imread(img)

        image_resize = tf.image.resize_with_pad(frame, 256, 256)

        image_input = tf.expand_dims(tf.cast(image_resize, dtype=tf.int32), axis=0)
        output = movenet(image_input)
        keypoints = output['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
        bboxes = output['output_0'].numpy()[:, :, 51:].reshape((6, 5))

        height, width, _ = frame.shape

        keypoints_image = draw(frame.copy(), keypoints, bboxes, result_size=(int(width / height * 512), 512))
        # keypoints_image = draw(frame.copy(), keypoints, bboxes, (512, 512))
        print(keypoints_image.shape)

        cv2.imshow('Camera Test', keypoints_image)

        cv2.waitKey()


if __name__ == '__main__':
    main()
