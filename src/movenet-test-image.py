import cv2
import os
from glob import glob
from lib.draw import draw
from lib.inference import load_movenet, forwarding_movenet, split_keypoints_bboxes


def main():
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    image_path = current_file_path + '/../test_file/image'
    images = glob(image_path + '/*.jpg')

    movenet = load_movenet()

    for img in images:
        frame = cv2.imread(img)

        keypoints, bboxes = split_keypoints_bboxes(forwarding_movenet(movenet, frame))

        height, width, _ = frame.shape

        keypoints_image = draw(frame.copy(), keypoints, bboxes, result_size=(int(width / height * 512), 512))
        print(keypoints_image.shape)

        cv2.imshow('Camera Test', keypoints_image)

        cv2.waitKey()


if __name__ == '__main__':
    main()
