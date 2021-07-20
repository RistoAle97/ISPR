import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, k_size, k_x, k_y, name):
        self.k_size = k_size
        self.k_x = k_x
        self.k_y = k_y
        self.name = name

    @abstractmethod
    def kernel_pad(self, m: np.ndarray, pad_zero: bool):
        if pad_zero:
            p = np.pad(m, [1, 1], mode='constant')
        else:
            p = np.concatenate((m[0, :].reshape((1, m.shape[1])), m, m[-1, :].reshape((1, m.shape[1]))), 0)
            p = np.concatenate((p[:, 0].reshape((p.shape[0]), 1), p, p[:, -1].reshape((p.shape[0]), 1)), 1)

        return p


class RobertsKernel(Kernel):

    def __init__(self):
        k_x = np.array(([1, 0], [0, -1]))
        k_y = np.array(([0, -1], [1, 0]))
        super(RobertsKernel, self).__init__(2, k_x, k_y, "Roberts_Kernel")

    def kernel_pad(self, m, pad_zero: bool):
        if pad_zero:
            padded_m = np.pad(m, [(0, 1), (0, 1)], mode='constant')
        else:
            padded_m = np.concatenate((m, m[-1, :].reshape(1, m.shape[1])), 0)
            padded_m = np.concatenate((padded_m, padded_m[:, -1].reshape((padded_m.shape[0]), 1)), 1)

        return padded_m


class PrewittKernel(Kernel):

    def __init__(self):
        k_size = 3
        k_x = np.array(([1, 0, -1], [1, 0, -1], [1, 0, -1]))
        k_y = k_x.T
        super(PrewittKernel, self).__init__(k_size, k_x, k_y, "Prewitt_Kernel")

    def kernel_pad(self, m, pad_zero: bool):
        return super(PrewittKernel, self).kernel_pad(m, pad_zero)


class SobelKernel(Kernel):

    def __init__(self):
        k_size = 3
        k_x = np.array(([1, 0, -1], [1, 0, -1], [1, 0, -1]))
        k_x[1, :] *= 2
        k_y = k_x.T
        super(SobelKernel, self).__init__(k_size, k_x, k_y, "Sobel_Kernel")

    def kernel_pad(self, m, pad_zero: bool):
        return super(SobelKernel, self).kernel_pad(m, pad_zero)
