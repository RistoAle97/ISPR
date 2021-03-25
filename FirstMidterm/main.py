# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from ISPR.kernel import *
import cv2
from ISPR.edgedetector import *
from ISPR.image_processor import *
import matplotlib.pyplot as plt
from PIL import Image


# def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def test_prewitt():
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    img_x = cv2.filter2D(Gm, -1, kernelx)
    img_y = cv2.filter2D(Gm, -1, kernely)
    R = np.concatenate((img_x, img_y))
    Image.fromarray(R).show()


def test_sobel():
    kernelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_x = cv2.filter2D(Gm, -1, kernelx)
    img_y = cv2.filter2D(Gm, -1, kernely)
    R = np.concatenate((img_x, img_y))
    Image.fromarray(R).show()


def test_roberts():
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, -1], [1, 0]])
    img_x = cv2.filter2D(Gm, -1, kernelx)
    img_y = cv2.filter2D(Gm, -1, kernely)
    plot = np.concatenate((img_x, img_y))
    Image.fromarray(plot).show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    r_kernel = RobertsKernel()
    p_kernel = PrewittKernel()
    s_kernel = SobelKernel()
    kernels = list()
    kernels.extend((r_kernel, p_kernel, s_kernel))
    image = "BMaps/2_29_s.bmp"
    Gm = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    edge_filter = EdgeDetector()
    i_kernel = NormalizedBlur()
    image_processors = list()
    # image_processors.extend((NormalizedBlur(), GaussianBlur(), Sharpener()))
    for kernel in kernels:
        i_x, i_y, m = edge_filter.compute(image, kernel, False, NormalizedBlur())
        edge_filter.plot_filtered_image(i_x, i_y, m)

    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
