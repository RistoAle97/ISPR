import keras
from ISPR.SecondMidterm.RBM import *
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


class Model:

    model_id = 0

    def __init__(self, encoder: list, classifier: list, name=None):
        if len(encoder) < 2 or not classifier:
            raise Exception("The encoder must have at least two dimensions and the classifier one")

        self.architecture = ""
        self.encoder = list()
        if encoder[0] is not RBM:
            for i in range(len(encoder)-1):
                self.encoder.append(RBM(encoder[i], encoder[i+1]))
                self.architecture += "RBM({0}, {1}) --> ".format(encoder[i], encoder[i+1])

            '''if len(encoder) != 2:
                self.encoder.append(RBM(encoder[-2], encoder[-1]))
                self.architecture += "RBM({0}, {1}) --> ".format(encoder[-2], encoder[-1])'''

        self.classifier = keras.models.Sequential()
        input_size = self.encoder[-1].weights.shape[0]
        if len(classifier) > 1:
            self.classifier.add(keras.layers.Dense(classifier[0], activation="relu", input_dim=input_size))
            self.architecture += "ReLu({0}) --> ".format(classifier[0])
            for layer in classifier[1:-1]:
                self.classifier.add(keras.layers.Dense(layer, activation="relu"))
                self.architecture += "ReLu({0}) --> ".format(layer)

            input_size = None

        self.classifier.add(keras.layers.Dense(classifier[-1], activation="softmax", input_dim=input_size))
        self.architecture += "SoftMax({0})".format(classifier[-1])
        if not name:
            name = "Model_"+str(Model.model_id)
            Model.model_id += 1

        self.name = name
        self.architecture = name + ": " + self.architecture

    def train_encoder(self, tr_set, eta: list, alpha: list, epochs: int, save_weights: bool = None):
        encodings = np.copy(tr_set)
        for i in range(len(self.encoder)):
            self.encoder[i].train(encodings, eta[i], alpha[i], epochs, save_weights)
            if 1 < len(self.encoder) != i:
                encodings = self.encoder[i].encode(encodings, False)

    def train_classifier(self, tr_encodings, tr_labels, epochs: int):
        tr_labels_one_hot = keras.utils.to_categorical(tr_labels)
        self.classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.classifier.fit(tr_encodings, tr_labels_one_hot, epochs=epochs)

    def encode(self, patterns, load_weights):
        encodings = np.copy(patterns)
        for e in self.encoder:
            encodings = e.encode(encodings, load_weights)

        return encodings

    def train(self, tr_set, tr_labels, rbm_eta: list, rbm_alpha: list, rbm_epochs, save_w, classifier_epochs):
        self.train_encoder(tr_set, rbm_eta, rbm_alpha, rbm_epochs, save_w)
        encodings = self.encode(tr_set, False)
        self.train_classifier(encodings, tr_labels, classifier_epochs)

    def predict_evaluate(self, patterns, values):
        encodings = self.encode(patterns, False)
        values_one_hot = keras.utils.to_categorical(values)
        return self.classifier.predict(encodings), self.classifier.evaluate(encodings, values_one_hot, verbose=0)

    @staticmethod
    def scores(labels_one_hot, predictions, print_scores: bool, averages: bool):
        classes = np.argmax(predictions, axis=1)
        if averages:
            average_value = None
        else:
            average_value = "macro"

        precision = precision_score(labels_one_hot, classes, average=average_value)
        recall = recall_score(labels_one_hot, classes, average=average_value)
        f_score = f1_score(labels_one_hot, classes, average=average_value)
        if print_scores:
            for p, r, f, i in zip(precision, recall, f_score, np.arange(len(precision))):
                print("Value: {0}, Precision: {1}, Recall: {2}, F1: {3}".format(i, p, r, f))
        return precision, recall, f_score
