from keras.datasets import cifar10
import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    (tr_set, tr_labels), (ts_set, ts_labels) = cifar10.load_data()
    tr_set = tr_set.astype("float32") / 255
    ts_set = ts_set.astype("float32") / 255
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=[32, 32, 3]))
    model.add(keras.layers.Conv2D(32, 3, padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D(2, 2, padding="same"))
    model.add(keras.layers.Conv2D(32, 3, padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(32, 3, padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D(2, 2, padding="same"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    tr_labels_one_hot = keras.utils.to_categorical(tr_labels)
    model.fit(tr_set, tr_labels_one_hot, epochs=30, workers=8, use_multiprocessing=True)
    classes_tr = np.argmax(model.predict(tr_set), axis=1)
    values = model.evaluate(tr_set, tr_labels_one_hot, verbose=0)
    print("Training Error: {0}, Training accuracy: {1}".format(values[0], values[1]))
    confusion_matrix_tr = confusion_matrix(tr_labels, classes_tr)
    ConfusionMatrixDisplay(confusion_matrix_tr).plot()
    plt.title("Confusion Matrix Training")
    plt.show()

    ts_labels_one_hot = keras.utils.to_categorical(ts_labels)
    classes_ts = np.argmax(model.predict(tr_set), axis=1)
    values = model.evaluate(ts_set, ts_labels_one_hot, verbose=0)
    print("Test Error: {0}, Test accuracy: {1}".format(values[0], values[1]))
    confusion_matrix_ts = confusion_matrix(ts_labels, classes_ts)
    ConfusionMatrixDisplay(confusion_matrix_ts).plot()
    plt.title("Confusion Matrix Test")
    plt.show()
