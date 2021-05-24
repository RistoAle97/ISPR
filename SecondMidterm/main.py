import keras
from keras.datasets import mnist
# import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sn
from SecondMidterm.RBM import *
# import cv2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from SecondMidterm.mnist_recognition import *
from scipy.ndimage.interpolation import rotate
from PIL import Image


def build_model():
    m = keras.models.Sequential()
    m.add(
        keras.layers.Dense(150, activation="relu", input_dim=100, kernel_regularizer=keras.regularizers.l2(0.0005)))
    m.add(keras.layers.Dense(80, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0005)))
    m.add(keras.layers.Dense(10, activation="softmax"))
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m


if __name__ == '__main__':
    (tr_set, tr_labels), (ts_set, ts_labels) = mnist.load_data()
    tr_set = tr_set.astype("float32")/255
    ts_set = ts_set.astype("float32")/255
    ts_set_rotate = np.copy(ts_set)
    tr_set = np.array([np.reshape(pattern, 784) for pattern in tr_set])
    ts_set = np.array([np.reshape(pattern, 784) for pattern in ts_set])
    rbm = RBM(tr_set.shape[1], 100)
    # rbm.train(tr_set, 0.1, 0.5, 5, 100, True)
    tr_encodings = rbm.encode(tr_set, True)
    tr_labels_one_hot = keras.utils.to_categorical(tr_labels)
    ts_labels_one_hot = keras.utils.to_categorical(ts_labels)
    # model = build_model()
    # model.fit(tr_encodings, tr_labels_one_hot, batch_size=128, epochs=100, workers=8, use_multiprocessing=True)
    # model.save("mnist_classifier.h5")
    model = keras.models.load_model("mnist_classifier.h5")
    model.summary()
    mnist_detector = MnistLiveRecognition(rbm, model)
    # mnist_detector.live_recognition()

    out_train = model.predict(tr_encodings)
    classes_tr = np.argmax(out_train, axis=1)
    values = model.evaluate(tr_encodings, tr_labels_one_hot, verbose=0)
    print("Training Error: {0}, Training accuracy: {1}".format(values[0], values[1]))
    confusion_matrix_tr = confusion_matrix(tr_labels, classes_tr)
    precision = precision_score(tr_labels, classes_tr, average=None)
    recall = recall_score(tr_labels, classes_tr, average=None)
    f_score = f1_score(tr_labels, classes_tr, average=None)

    ts_encodings = rbm.encode(ts_set, False)
    out_test = model.predict(ts_encodings)
    classes_ts = np.argmax(out_test, axis=1)
    values = model.evaluate(ts_encodings, ts_labels_one_hot, verbose=0)
    print("Test Error: {0}, Test accuracy: {1}".format(values[0], values[1]))
    confusion_matrix_ts = confusion_matrix(ts_labels, classes_ts)
    precision_ts = precision_score(ts_labels, classes_ts, average=None)
    recall_ts = recall_score(ts_labels, classes_ts, average=None)
    f_score_ts = f1_score(ts_labels, classes_ts, average=None)

    # Image.fromarray(ts_set_rotate[9]*255).save("mnist.png")
    # Image.fromarray(rotate(ts_set_rotate[9]*255, 30, reshape=False)).save("mnistrotate.png")
    # Image.fromarray(rotate(ts_set_rotate[9]*255, -30, reshape=False)).save("mnistrotate1.png")
    cv2.imwrite("mnist.png", ts_set_rotate[9]*255)
    cv2.imwrite("mnistrotate.png", rotate(ts_set_rotate[9]*255, 30, reshape=False))
    cv2.imwrite("mnistrotate1.png", rotate(ts_set_rotate[9]*255, -30, reshape=False))
    ts_set_rotate = [rotate(pattern, np.random.randint(-40, 40), reshape=False) for pattern in ts_set_rotate]
    ts_set_rotate = np.array([np.reshape(pattern, 784) for pattern in ts_set_rotate])
    rotate_encodings = rbm.encode(ts_set_rotate, False)
    out_rotate = model.predict(rotate_encodings)
    classes_rotate = np.argmax(out_rotate, axis=1)
    values_rotate = model.evaluate(rotate_encodings, ts_labels_one_hot, verbose=0)
    print("Test Rotated Error: {0}, Test Rotated accuracy: {1}".format(values_rotate[0], values_rotate[1]))
    confusion_matrix_ts_rotate = confusion_matrix(ts_labels, classes_rotate)

    '''ConfusionMatrixDisplay(confusion_matrix_tr).plot()
    plt.title("Confusion Matrix Training")
    plt.show()
    ConfusionMatrixDisplay(confusion_matrix_ts).plot()
    plt.title("Confusion Matrix Test")
    plt.show()'''
    ConfusionMatrixDisplay(confusion_matrix_ts_rotate).plot()
    plt.title("Confusion Matrix Test Rotated")
    plt.show()
