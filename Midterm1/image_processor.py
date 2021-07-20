import numpy as np
from abc import ABC, abstractmethod


class ImageProcessor(ABC):

    def __init__(self, size, kernel, name):
        self.size = size
        self.kernel = kernel
        self.name = name

    '''@abstractmethod'''
    @staticmethod
    def image_pad(m: np.ndarray, pad_zero: bool):
        if pad_zero:
            p = np.pad(m, [1, 1], mode='constant')
        else:
            p = np.concatenate((m[0, :].reshape((1, m.shape[1])), m, m[-1, :].reshape((1, m.shape[1]))), 0)
            p = np.concatenate((p[:, 0].reshape((p.shape[0]), 1), p, p[:, -1].reshape((p.shape[0]), 1)), 1)

        return p

    def apply(self, image):
        image = self.image_pad(image, False)
        i_x = np.zeros((image.shape[0] - self.size + 1, image.shape[1] - self.size + 1))
        for i in range(len(i_x)):
            for j in range(len(i_x[0])):
                i_x[i, j] = np.sum(np.multiply(image[i:i + self.size, j:j + self.size], self.kernel))

        return i_x


class NormalizedBlur(ImageProcessor):

    def __init__(self):
        # size = 3
        kernel = np.ones((3, 3))/9
        super(NormalizedBlur, self).__init__(3, kernel, "Normalized_Blur")


class GaussianBlur(ImageProcessor):

    def __init__(self):
        kernel = np.ones((3, 3))
        kernel[1, :] *= 2
        kernel[:, 1] *= 2
        kernel /= 16
        super(GaussianBlur, self).__init__(3, kernel, "Gaussian_Blur")


class Sharpener(ImageProcessor):

    def __init__(self):
        kernel = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]))
        super(Sharpener, self).__init__(3, kernel, "Sharpener")
