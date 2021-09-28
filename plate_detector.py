import shutil

import cv2
import numpy as np
import tensorflow as tf


class Detector:

    def __init__(self):
        self._IMAGE_SIZE = 224
        self._model = tf.keras.models.load_model('my_model.h5')

    def detect(self, file):
        image = np.asarray(bytearray(file.file.read()), dtype="uint8")
        image = cv2.imdecode(image, flags=cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self._IMAGE_SIZE, self._IMAGE_SIZE))
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        p = self._model.predict(image)
        return p[0] * 255
