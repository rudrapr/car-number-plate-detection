import cv2
import numpy as np
import tensorflow as tf


class Detector:

    def __init__(self):
        self._IMAGE_SIZE = 224

        self.interpreter = tf.lite.Interpreter(model_path="./models/model_f16_quant.tflite")
        self.interpreter.allocate_tensors()

    def detect(self, file):
        image = np.asarray(bytearray(file.file.read()), dtype="uint8")
        image = cv2.imdecode(image, flags=cv2.IMREAD_COLOR)

        image = cv2.resize(image, (self._IMAGE_SIZE, self._IMAGE_SIZE))
        original = image
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]['index'], image)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        output_data = output_data[0] * 255

        x2 = int(output_data[0])
        y2 = int(output_data[1])
        x1 = int(output_data[2])
        y1 = int(output_data[3])

        ROI = original[y1:y2, x1:x2]
        return output_data, ROI
