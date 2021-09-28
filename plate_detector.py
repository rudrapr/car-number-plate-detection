import cv2
import numpy as np
import tensorflow as tf


class Detector:

    def __init__(self):
        self._IMAGE_SIZE = 224

        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path="model.tflite")
        self.interpreter.allocate_tensors()

    def detect(self, file):
        image = np.asarray(bytearray(file.file.read()), dtype="uint8")
        image = cv2.imdecode(image, flags=cv2.IMREAD_COLOR)
        print('decoded')
        image = cv2.resize(image, (self._IMAGE_SIZE, self._IMAGE_SIZE))
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

        self.interpreter.set_tensor(input_details[0]['index'], image)

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        print('predicted')
        return output_data[0] * 255
