# import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from ISPR.SecondMidterm.RBM import *
import time

if __name__ == '__main__':
    (tr_set, tr_labels), (ts_set, ts_labels) = mnist.load_data()
    plt.plot(tr_set[0])
    # plt.show()
    tr_set = tr_set.astype("float32")/255
    ts_set = ts_set.astype("float32")/255
    tr_set = np.array([np.reshape(pattern, 784) for pattern in tr_set])
    ts_set = np.array([np.reshape(pattern, 784) for pattern in ts_set])
    rbm = RBM(tr_set.shape[1], 100)
    t = time.time()
    rbm.train(tr_set, 0.3, 5, True)
    print(time.time() - t)
    encodings = rbm.encode(tr_set, False)
