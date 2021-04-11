import numpy as np


class RBM:

    def __init__(self, n_visible, n_hidden):
        self.weights = np.random.uniform(size=(n_hidden, n_visible))
        self.visibile_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)
        # pass

    @staticmethod
    def __clamp_data(x):
        # y = np.zeros(np.shape(x))
        return np.where(x > np.random.rand(np.size(x)), 1, 0)
        # return y

    def train(self, tr_set, eta, epochs):
        for i in range(epochs):
            for pattern in tr_set:
                # wake
                clamped_pattern = self.__clamp_data(pattern)
                hidden_nets = np.dot(self.weights, clamped_pattern) + self.hidden_bias
                hidden_p = 1/(1+np.exp(-hidden_nets))
                wake = np.outer(hidden_p, pattern)

                # dream
                stochastic_hidden_p = self.__clamp_data(hidden_p)
                visible_nets = np.dot(self.weights.T, stochastic_hidden_p) + self.visibile_bias
                visible_p = 1/(1+np.exp(-visible_nets))
                stochastic_reconstruction = self.__clamp_data(visible_p)
                negative_hidden_p = 1/(1 + np.exp(-np.dot(self.weights,stochastic_reconstruction)-self.hidden_bias))
                dream = np.outer(negative_hidden_p, stochastic_reconstruction)

                self.weights += eta * (wake - dream)/len(tr_set)
                self.visibile_bias += eta * (np.sum(hidden_p) - np.sum(negative_hidden_p))/len(tr_set)
                self.hidden_bias += eta * (np.sum(clamped_pattern) - np.sum(stochastic_reconstruction))/len(tr_set)
                break
            break
