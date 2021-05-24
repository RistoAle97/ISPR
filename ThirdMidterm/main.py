from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


def build_model():
    model = Sequential()
    model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=[32, 32, 3]))
    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2, padding="same"))
    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2, padding="same"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def run_model(model, tr_ts_set, tr_ts_labels, set_labels: str, plot_conf_matrix: bool):
    classes_labels = np.argmax(model.predict(tr_ts_set), axis=1)
    values = model.evaluate(tr_ts_set, tr_ts_labels, verbose=0)
    print("{0} Error: {1}, {0} accuracy: {2}".format(set_labels, values[0], values[1]))
    if plot_conf_matrix:
        conf_matrix = confusion_matrix(tr_ts_labels, classes_labels)
        ConfusionMatrixDisplay(conf_matrix).plot()
        plt.title("Confusion Matrix {0}".format(set_labels))
        plt.show()


if __name__ == '__main__':
    (tr_set, tr_labels), (ts_set, ts_labels) = cifar10.load_data()
    tr_set = tr_set.astype("float32") / 255
    ts_set = ts_set.astype("float32") / 255
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    tr_labels_one_hot = to_categorical(tr_labels)
    ts_labels_one_hot = to_categorical(ts_labels)

    m = build_model()
    m.fit(tr_set, tr_labels_one_hot, epochs=20, workers=8, use_multiprocessing=True)
    run_model(m, tr_set, tr_labels_one_hot, "Training", True)
    run_model(m, ts_set, ts_labels_one_hot, "Test", True)
