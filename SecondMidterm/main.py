# import keras
from keras.datasets import mnist
import numpy as np
import matplotlib
from ISPR.SecondMidterm.RBM import *

if __name__ == '__main__':
    (tr_set, tr_labels), (ts_set, ts_labels) = mnist.load_data()
    tr_set = tr_set.astype("float32")/255
    ts_set = ts_set.astype("float32")/255
    tr_set = np.array([np.reshape(pattern, 784) for pattern in tr_set])
    ts_set = np.array([np.reshape(pattern, 784) for pattern in ts_set])
    rbm = RBM(tr_set.shape[1], 100)
    rbm.train(tr_set, 0.01, 1, True)
