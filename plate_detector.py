import cv2
import numpy as np
import tensorflow as tf


class Detector:

    def __init__(self):
        self._IMAGE_SIZE = 224
        self._model = tf.keras.models.load_model('my_model.h5')

    def detect(self, file):
        image = cv2.imread('page2_2.jpg')
        image = cv2.resize(image, (self._IMAGE_SIZE, self._IMAGE_SIZE))
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        p = self._model.predict(image)
        return p[0] * 255
