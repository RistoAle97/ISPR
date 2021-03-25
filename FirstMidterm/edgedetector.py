import numpy as np
import cv2
from PIL import Image


class EdgeDetector:

    def __init__(self):
        pass

    @staticmethod
    def __convolution(image, kernel):
        k = kernel.k_size
        i_x = np.zeros((image.shape[0] - k + 1, image.shape[1] - k + 1))
        i_y = np.zeros((image.shape[0] - k + 1, image.shape[1] - k + 1))
        for i in range(len(i_x)):
            for j in range(len(i_x[0])):
                i_x[i, j] = np.sum(np.multiply(image[i:i + k, j:j + k], kernel.k_x))
                i_y[i, j] = np.sum(np.multiply(image[i:i + k, j:j + k], kernel.k_y))

        return i_x, i_y

    def compute(self, image_path, kernel, pad_zero: bool, image_processor):
        image_gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        if image_processor is not None:
            image_gray = image_processor.apply(image_gray)

        image_gray = kernel.kernel_pad(image_gray, pad_zero)
        i_x, i_y = self.__convolution(image_gray, kernel)
        magnitude = np.sqrt(i_x**2+i_y**2)
        return i_x, i_y, magnitude

    @staticmethod
    def plot_filtered_image(i_x, i_y, magnitude):
        image = np.concatenate((i_x, i_y, magnitude), 0)
        Image.fromarray(image).show()
