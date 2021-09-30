import string

import cv2
import numpy as np
import tensorflow as tf


class OCR:

    def __init__(self):
        self._IMAGE_SIZE = (200, 31)
        self.alphabets = string.digits + string.ascii_lowercase

        self.interpreter = tf.lite.Interpreter(model_path='./models/ocr_f16.tflite')
        self.interpreter.allocate_tensors()

    def recognize(self, ROI):
        input_data = cv2.resize(ROI, self._IMAGE_SIZE)
        input_data = np.dot(input_data[..., :3], [0.2989, 0.5870, 0.1140])

        input_data = input_data[np.newaxis]
        input_data = np.expand_dims(input_data, 3)
        input_data = input_data.astype('float32') / 255

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(output_details[0]['index'])

        blank_index = len(self.alphabets)
        final_output = "".join(self.alphabets[index] for index in output[0] if index not in [blank_index, -1])

        return final_output
