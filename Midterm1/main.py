from kernel import *
from edgedetector import *
from image_processor import *


if __name__ == '__main__':
    r_kernel = RobertsKernel()
    p_kernel = PrewittKernel()
    s_kernel = SobelKernel()
    kernels = list()
    kernels.extend((r_kernel, p_kernel, s_kernel))
    images = list()
    images.extend(("BMaps/2_29_s.bmp", "BMaps/6_14_s.bmp"))
    edge_filter = EdgeDetector()
    image_processors = list()
    image_processors.extend((None, NormalizedBlur(), GaussianBlur(), Sharpener()))
    for image in images:
        for kernel in kernels:
            for p in image_processors:
                i_x, i_y, m = edge_filter.compute(image, kernel, False, p)
                i_map = image.replace("BMaps/", "")
                i_map = i_map.replace(".bmp", "")
                edge_filter.plot_filtered_image(i_x, i_y, m, i_map, kernel, p)
