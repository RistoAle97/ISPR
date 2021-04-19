import numpy as np
import tqdm
import csv


class RBM:

    rbm_id = 0

    def __init__(self, n_visible, n_hidden):
        self.weights = np.random.normal(0, 0.1, (n_hidden, n_visible))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)
        self.name = "RBM_"+str(RBM.rbm_id)
        RBM.rbm_id += 1

    @staticmethod
    def __clamp_data(x):
        return np.where(x > np.random.rand(*x.shape), 1, 0)  # .astype("float32")

    @staticmethod
    def __sigmoid(x):
        return 1/(1 + np.exp(-x))

    def __save_weights(self, file: str):
        try:
            with open(file, "w", newline="") as datafile:
                w_writer = csv.writer(datafile, delimiter=",")
                for w in self.weights:
                    w_writer.writerow(w)

                w_writer.writerows([["# Visible Bias"], self.visible_bias, ["# Hidden Bias"], self.hidden_bias])

        except IOError:
            print("It was impossible to retrieve or write data on file " + file)

    def __load_weights(self, file: str):
        try:
            with open(file) as datafile:
                filereader = csv.reader(datafile, delimiter=",")
                for row, i in zip(filereader, range(len(self.weights))):
                    self.weights[i] = row

                hidden = False
                for row in filereader:
                    if (filereader.line_num - len(self.weights)) % 2 == 1:
                        continue

                    if not hidden:
                        self.visible_bias = np.array(row).astype("float32")
                        hidden = True
                    else:
                        self.hidden_bias = np.array(row).astype("float32")

        except IOError:
            print("It was impossible to retrieve or read data on file " + file)

    def __contrastive_divergence_step(self, values, step: str):
        bin_values = self.__clamp_data(values)
        if step == "pos_data" or step == "neg_data":
            nets = np.dot(self.weights, bin_values) + self.hidden_bias
            nodes_p = self.__sigmoid(nets)
            if step == "pos_data":
                data = np.outer(nodes_p, values)
                return nodes_p, data
            else:
                data = np.outer(nodes_p, bin_values)
                return bin_values, nodes_p, data
        else:
            nets = np.dot(self.weights.T, bin_values) + self.visible_bias
            nodes_p = self.__sigmoid(nets)
            return nodes_p

    def train(self, tr_set, eta, alpha, epochs: int, batch_size: int, save_weights: bool = False):
        tr_set_copy = np.copy(tr_set)
        for i in range(epochs):
            np.random.shuffle(tr_set_copy)
            for pattern in tqdm.tqdm(tr_set, desc="Training {0}, Epoch {1}".format(self.name, i)):
                h_p, wake = self.__contrastive_divergence_step(pattern, "pos_data")
                v_p = self.__contrastive_divergence_step(h_p, "reconstruction")
                bin_reconstruction, neg_h_p, dream = self.__contrastive_divergence_step(v_p, "neg_data")

                self.weights += eta * (wake - dream) / batch_size  # len(tr_set)
                self.visible_bias += eta * (np.sum(pattern) - np.sum(bin_reconstruction)) / batch_size
                self.hidden_bias += eta * (np.sum(h_p) - np.sum(neg_h_p)) / batch_size

        if save_weights:
            self.__save_weights("weights_{0}.csv".format(self.name))

    def encode(self, patterns, load_weigths: bool):
        if load_weigths:
            self.__load_weights("weights_{0}.csv".format(self.name))

        y = list()
        bin_patterns = self.__clamp_data(patterns)
        for bin_pattern in bin_patterns:
            h_nets = np.dot(self.weights, bin_pattern) + self.hidden_bias
            y.append(self.__sigmoid(h_nets))

        return np.array(y)

    def reconstruction(self, patterns, load_weights: bool):
        if load_weights:
            self.__load_weights("weights_{0}.csv".format(self.name))

        y = list()
        for pattern in patterns:
            h_p, wake = self.__contrastive_divergence_step(pattern, "pos_data")
            y.append(self.__contrastive_divergence_step(h_p, "reconstruction"))

        return np.array(y)
