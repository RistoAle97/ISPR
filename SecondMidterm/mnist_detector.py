import cv2
# import numpy as np
import keras
from ISPR.SecondMidterm.RBM import *
# from PIL import Image


class MnistDetector:

    def __init__(self, rbm: RBM, model: keras.models):
        self.rbm = rbm
        self.model = model

    def live_recognition(self):
        cp = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cp.read()
            cv2.rectangle(frame, (0, 366), (84, 478), (0, 255, 0), thickness=2)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, blockSize=321, C=28)

            image = frame_gray[366:478, 0:84]
            resized_image = cv2.resize(image, (28, 28)).astype("float32")/255
            image_pattern = np.array(np.reshape(resized_image, 784))
            encoded_image = self.rbm.encode(image_pattern, True)
            out = self.model.predict(encoded_image)
            predicted_value = np.argmax(out, axis=1)

            cv2.putText(frame, "Model prediction: "+str(predicted_value), (210, 459),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 0, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA)

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) == ord("q"):
                break
            # elif cv2.waitKey(1) == ord("i"):
            #    self.is_inferecing = not self.is_inferecing

        cp.release()
