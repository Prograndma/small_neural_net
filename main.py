import numpy as np
from arff import Arff
import os
from sklearn.neural_network import MLPClassifier as skMLP
from sklearn.model_selection import GridSearchCV
from mlp import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'

from sklearn.datasets import load_boston



def main():
    mat = Arff(f"{dir_path}vowels.arff")
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    one_hot = np.zeros((data.shape[0], len(np.unique(labels))))
    for i, row in enumerate(labels.astype(int)):
        one_hot[i, row] += 1
    data = data[:, 3:]

    labels = one_hot

    model = MLPClassifier(momentum=.5, valid_size=.25)
    MSE_valid, accuracies, MSE_train, num_epochs, MSE_test = model.fit(data, labels)

    plt.plot(accuracies, label="Accuracy of model")
    plt.plot(MSE_train, label="MSE on Training set")
    plt.plot(MSE_valid, label="MSE on Validation set")
    plt.plot(MSE_test, label="MSE on Testing set")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()


if __name__ == "__main__":
    main()
    exit()
