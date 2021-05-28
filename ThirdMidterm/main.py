from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, BatchNormalization
from keras import regularizers, models
from keras.losses import MSE  # CategoricalCrossentropy, MSE
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import logging


def build_model():
    """model = Sequential()
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
    model.summary()"""
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), 2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), 2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def run_model(model, tr_ts_set, tr_ts_labels, tr_ts_labels_one_hot, set_labels: str, plot_conf_matrix: bool):
    classes_labels = np.argmax(model.predict(tr_ts_set), axis=1)
    values = model.evaluate(tr_ts_set, tr_ts_labels_one_hot, verbose=0)
    print("{0} Error: {1}, {0} accuracy: {2}".format(set_labels, values[0], values[1]))
    if plot_conf_matrix:
        conf_matrix = confusion_matrix(tr_ts_labels, classes_labels)
        ConfusionMatrixDisplay(conf_matrix).plot()
        plt.title("Confusion Matrix {0}".format(set_labels))
        plt.show()


def adversary_pattern(model, pattern, label, eps=2/255.0):
    pattern = tf.cast(np.reshape(pattern, (1, 32, 32, 3)), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(pattern)
        pred = model(pattern)
        loss = MSE(label, pred)
        gradient = tape.gradient(loss, pattern)
        signed_grad = tf.sign(gradient)
        adversary = (pattern + (signed_grad * eps)).numpy()
        return adversary


def attack_pattern(model, pattern, label, eps, predict: bool, print_prediction: bool):
    adversary_image = adversary_pattern(m, pattern, label, eps=eps)
    if predict:
        out_adversary = np.argmax(model.predict(adversary_image))
        out_pattern = np.argmax(model.predict(pattern.reshape(1, 32, 32, 3)))
        if print_prediction:
            print("True label: {0}, Predicted label: {1}, Predicted adversary label: {2}".
                  format(classes[np.argmax(label)], classes[out_pattern], classes[out_adversary]))

    adversary_image = adversary_image.reshape((32, 32, 3))
    adversary_image = np.clip(adversary_image * 255, 0, 255)
    image = np.copy(pattern) * 255
    plt.imshow(adversary_image.astype("uint8"))
    plt.show()
    plt.imshow(image.astype("uint8"))
    plt.show()


def add_noise_set(model, patterns, labels, size, eps):
    patterns_adversary_list = np.copy(patterns)
    if size != len(patterns):
        patterns_to_attack = np.random.choice(np.arange(len(patterns)), size)
    else:
        patterns_to_attack = np.arange(len(patterns))

    for i in patterns_to_attack:
        patterns_adversary_list[i] = adversary_pattern(model, patterns[i], labels[i], eps=eps).reshape(32, 32, 3)

    return np.array(patterns_adversary_list)


if __name__ == '__main__':
    (tr_set, tr_labels), (ts_set, ts_labels) = cifar10.load_data()
    tr_set = tr_set.astype("float32") / 255
    ts_set = ts_set.astype("float32") / 255
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    tr_labels_one_hot = to_categorical(tr_labels)
    ts_labels_one_hot = to_categorical(ts_labels)

    # m = build_model()
    # m.fit(tr_set, tr_labels_one_hot, epochs=10, workers=16, use_multiprocessing=True)
    m = models.load_model("cifar_classifier.h5")
    # m = models.load_model("my_model_2.h5")
    # run_model(m, tr_set, tr_labels, tr_labels_one_hot, "Training", True)
    # run_model(m, ts_set, ts_labels, ts_labels_one_hot, "Test", True)

    eps_attack = 0.001
    attack_pattern(m, ts_set[0], ts_labels_one_hot[0], eps_attack, True, True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    ts_set_adversary = add_noise_set(m, ts_set, ts_labels_one_hot, len(ts_set), eps_attack)
    run_model(m, ts_set_adversary, ts_labels, ts_labels_one_hot, "Test adversary eps={0}".format(eps_attack), True)
