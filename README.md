# MoveNet Test Repository

- [About MoveNet][movenet]
- [MoveNet Multipose Pretrained Model(Tensorflow hub)][hub-model]

## Getting started

Requirements:

- tensorflow
- tensorflow-hub
- opencv-python
- numpy
- pandas
- matplotlib
- scikit-learn
- pygame
- cuda, cudnn (in case of using nvidia gpu)

Just run following command to get above packages:

```sh
pip install tensorflow tensorflow-hub opencv-python numpy pandas matplotlib scikit-learn pygame
```

Or, using requirements.txt:

```sh
pip install -r requirements.txt
```

And now you can run the following command(with webcam).

```sh
python src/stop-dumping.py
```

Or with RTSP video.

```sh
python src/stop-dumping.py -u rtsp://your-rtsp-url.com:554
```

[movenet]: https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html
[hub-model]: https://tfhub.dev/google/movenet/multipose/lightning/1