# MoveNet Test Repository

- [About MoveNet][movenet]
- [MoveNet Multipose Pretrained Model(Tensorflow hub)][hub-model]

## Getting started

Requirements:

- tensorflow
- tensorflow-hub
- opencv-python
- numpy

Just run following command to get above packages:

```sh
pip install tensorflow tensorflow-hub opencv-python numpy
```

Or, using requirements.txt:

```sh
pip install -r requirements.txt
```

And now you can run the following command.

```sh
python src/movenet-test-webcam.py
```

[movenet]: https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html
[hub-model]: https://tfhub.dev/google/movenet/multipose/lightning/1