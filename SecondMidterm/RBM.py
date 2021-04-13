import numpy as np
import csv


class RBM:

    def __init__(self, n_visible, n_hidden):
        self.weights = np.random.normal(0, 0.01, (n_hidden, n_visible))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

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
                # w_writer.writerow(self.visibile_bias)
                # w_writer.writerows(["# Hidden Bias"])
                # w_writer.writerow(self.hidden_bias)

        except IOError:
            print("It was impossible to retrieve or write data on file " + file)

    def __load_weights(self, file: str):
        try:
            with open(file) as datafile:
                filereader = csv.reader(datafile, delimiter=",")
                hidden = False
                bias = False
                for row, i in zip(filereader, range(len(self.weights))):
                    if row[0][0] == "#":
                        bias = True
                        # continue
                    elif not hidden and bias:
                        self.visible_bias = row
                        hidden = True
                    elif hidden and bias:
                        self.hidden_bias = row
                    else:
                        self.weights[i] = row

        except IOError:
            print("It was impossible to retrieve or read data on file " + file)

    def train(self, tr_set, eta, epochs: int, save_weights: bool):
        # self.__load_weights("weights.csv")
        for i in range(epochs):
            bin_tr_set = self.__clamp_data(tr_set)
            for pattern, bin_pattern in zip(tr_set, bin_tr_set):
                # wake
                # clamped_pattern = self.__clamp_data(pattern)
                # hidden_nets = np.dot(self.weights, binary_pattern) + self.hidden_bias
                h_nets = np.dot(self.weights, bin_pattern) + self.hidden_bias
                # hidden_p = 1/(1+np.exp(-hidden_nets))  # .astype("float32")
                h_p = self.__sigmoid(h_nets)
                # wake = np.outer(hidden_p, pattern)
                wake = np.outer(h_p, pattern)

                # dream
                # stochastic_hidden_p = self.__clamp_data(hidden_p)
                bin_h_p = self.__clamp_data(h_p)
                # visible_nets = np.dot(self.weights.T, stochastic_hidden_p) + self.visibile_bias
                v_nets = np.dot(self.weights.T, bin_h_p) + self.visible_bias
                # visible_p = 1/(1+np.exp(-visible_nets))
                v_p = self.__sigmoid(v_nets)
                # stochastic_reconstruction = self.__clamp_data(visible_p)
                bin_reconstruction = self.__clamp_data(v_p)
                # negative_hidden_p = 1/(1 + np.exp(-np.dot(self.weights, stochastic_reconstruction)-self.hidden_bias))
                neg_h_nets = np.dot(self.weights, bin_reconstruction) + self.hidden_bias
                neg_h_p = self.__sigmoid(neg_h_nets)
                # dream = np.outer(negative_hidden_p, stochastic_reconstruction)
                dream = np.outer(neg_h_p, bin_reconstruction)

                self.weights += eta * (wake - dream)/len(tr_set)
                # self.visibile_bias += eta * (np.sum(hidden_p) - np.sum(negative_hidden_p))/len(tr_set)
                self.visible_bias += eta * (np.sum(h_p) - np.sum(neg_h_p)) / len(tr_set)
                # self.hidden_bias += eta * (np.sum(binary_pattern) - np.sum(stochastic_reconstruction))/len(tr_set)
                self.hidden_bias += eta * (np.sum(bin_pattern) - np.sum(bin_reconstruction)) / len(tr_set)
                # break
            # break
        if save_weights:
            self.__save_weights("weights.csv")
