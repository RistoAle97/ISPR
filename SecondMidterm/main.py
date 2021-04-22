import keras
from keras.datasets import mnist
# import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sn
from ISPR.SecondMidterm.RBM import *
import cv2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
# from ISPR.SecondMidterm.live_recognition import *

if __name__ == '__main__':
    (tr_set, tr_labels), (ts_set, ts_labels) = mnist.load_data()
    tr_set = tr_set.astype("float32")/255
    ts_set = ts_set.astype("float32")/255
    tr_set = np.array([np.reshape(pattern, 784) for pattern in tr_set])
    ts_set = np.array([np.reshape(pattern, 784) for pattern in ts_set])
    rbm = RBM(tr_set.shape[1], 100)
    # rbm.train(tr_set, 0.1, 0.5, 5, 100, True)
    tr_encodings = rbm.encode(tr_set, True)
    tr_labels_one_hot = keras.utils.to_categorical(tr_labels)
    ts_labels_one_hot = keras.utils.to_categorical(ts_labels)
    '''model = keras.models.Sequential()
    model.add(keras.layers.Dense(150, activation="relu", input_dim=100))
    model.add(keras.layers.Dense(80, activation="relu"))
    # model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(50, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(tr_encodings, tr_labels_one_hot, epochs=10, workers=8, use_multiprocessing=True)'''
    # model.save("mnist_classifier.h5")
    model = keras.models.load_model("mnist_classifier.h5")
    out_train = model.predict(tr_encodings)
    values = model.evaluate(tr_encodings, tr_labels_one_hot, verbose=0)
    print("Training Error: {0}, Training accuracy: {1}".format(values[0], values[1]))
    confusion_matrix_tr = confusion_matrix(tr_labels, np.argmax(out_train, axis=1))
    precision = precision_score(tr_labels, np.argmax(out_train, axis=1), average=None)
    recall = recall_score(tr_labels, np.argmax(out_train, axis=1), average=None)
    f_score = f1_score(tr_labels, np.argmax(out_train, axis=1), average=None)
    ts_encodings = rbm.encode(ts_set, False)
    out_test = model.predict(ts_encodings)
    values = model.evaluate(ts_encodings, ts_labels_one_hot, verbose=0)
    confusion_matrix_ts = confusion_matrix(ts_labels, np.argmax(out_test, axis=1))
    precision_ts = precision_score(ts_labels, np.argmax(out_test, axis=1), average=None)
    recall_ts = recall_score(ts_labels, np.argmax(out_test, axis=1), average=None)
    f_score_ts = f1_score(ts_labels, np.argmax(out_test, axis=1), average=None)
    print("Test Error: {0}, Test accuracy: {1}".format(values[0], values[1]))
    ConfusionMatrixDisplay(confusion_matrix_tr).plot(cmap="binary")
    plt.title("Confusion Matrix Training")
    plt.show()
    # plt.savefig("conf_matrix_tr.png")
    # plt.savefig("")
    ConfusionMatrixDisplay(confusion_matrix_ts).plot()
    plt.title("Confusion Matrix Test")
    plt.show()
