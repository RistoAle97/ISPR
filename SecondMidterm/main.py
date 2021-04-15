import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from ISPR.SecondMidterm.RBM import *
import time
# import sklearn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

if __name__ == '__main__':
    (tr_set, tr_labels), (ts_set, ts_labels) = mnist.load_data()
    plt.plot(tr_set[0])
    # plt.show()
    tr_set = tr_set.astype("float32")/255
    ts_set = ts_set.astype("float32")/255
    tr_set = np.array([np.reshape(pattern, 784) for pattern in tr_set])
    ts_set = np.array([np.reshape(pattern, 784) for pattern in ts_set])
    rbm = RBM(tr_set.shape[1], 100)
    # t = time.time()
    # rbm.train(tr_set, 0.1, 0.5, 5, True)
    # print(time.time() - t)
    '''reconstrunctions = rbm.reconstruction(tr_set, True)
    image = np.reshape(reconstrunctions[0]*255, (28, 28))
    plt.imshow(image)
    plt.show()
    plt.imshow(np.reshape(tr_set[0]*255, (28, 28)))
    plt.show()'''
    tr_encodings = rbm.encode(tr_set, True)
    tr_labels_one_hot = keras.utils.to_categorical(tr_labels)
    ts_labels_one_hot = keras.utils.to_categorical(ts_labels)
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(80, activation="relu", input_dim=100))
    model.add(keras.layers.Dense(50, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(tr_encodings, tr_labels_one_hot, epochs=10)
    out_train = model.predict(tr_encodings)
    values = model.evaluate(tr_encodings, tr_labels_one_hot, verbose=0)
    print("Training Error: {0}, Training accuracy: {1}".format(values[0], values[1]))
    # out_train = np.argmax(out_train, axis=1)
    confusion_matrix = confusion_matrix(tr_labels, np.argmax(out_train, axis=1))
    precision = precision_score(tr_labels, np.argmax(out_train, axis=1), average=None)
    recall = recall_score(tr_labels, np.argmax(out_train, axis=1), average=None)
    f_score = f1_score(tr_labels, np.argmax(out_train, axis=1), average=None)
    print("Training:\nPrecision: {0}, Recall: {1}, F1: {2}".format(precision, recall, f_score))
    ts_encodings = rbm.encode(ts_set, False)
    out_test = model.predict(ts_encodings)
    values = model.evaluate(ts_encodings, ts_labels_one_hot, verbose=0)
    print("Test Error: {0}, Test accuracy: {1}".format(values[0], values[1]))
    # out_test = model.predict(encodings)
