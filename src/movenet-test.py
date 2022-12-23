import tensorflow_hub as hub
import tensorflow as tf
from keras.models import Sequential
import numpy as np

model_url = 'https://tfhub.dev/google/movenet/multipose/lightning/1'
loaded = hub.load(model_url)
# keras_model = Sequential([
#     hub.KerasLayer(model_url,signature='serving_default', signature_outputs_as_dict=True, input_shape=(256, 256, 3))
# ])
keras_model = hub.KerasLayer(model_url,signature='serving_default', signature_outputs_as_dict=True, input_shape=(256, 256, 3))

keras_model.summary()

# print(np.sum([np.prod(v.get_shape().as_list()) for v in loaded.trainable_variables()]))
