import tensorflow as tf
import tensorflow_hub as hub


def load_movenet():
    model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    return model.signatures['serving_default']



def forwarding_movenet(movenet, image):
    image_resize = tf.image.resize_with_pad(image, 256, 256)

    image_input = tf.expand_dims(tf.cast(image_resize, dtype=tf.int32), axis=0)
    return movenet(image_input)


def split_keypoints_bboxes(output):
    return (output['output_0'].numpy()[:, :, :51].reshape((6, 17, 3)),
            output['output_0'].numpy()[:, :, 51:].reshape((6, 5)))
