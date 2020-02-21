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
    # mat = Arff(f"{dir_path}iris.arff")
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


    # boston = load_boston()
    # x = boston.data
    # y = boston.target
    # bad = skMLP()
    # bad.fit(data, labels)
    # print(f"bad_score = {bad.score(data, labels)}")
    # skitch = skMLP()
    # params = {'learning_rate_init' : [.001, .01, .1,],
    #           'hidden_layer_sizes' : [(1000), (100, 100, 100), (16, 16, 16, 16)],
    #           'early_stopping' : [True],
    #           'momentum' : [.5, .9, 1]}
    # grid_search = GridSearchCV(skitch, params, 'accuracy')
    # grid_search = grid_search.fit(data, labels)
    # score = grid_search.score(data, labels)
    # print(f"score = {score}")
    # print(labels)
    #
    # data = [[0,0],
    #         [0,1]]
    # targ = [[1],
    #         [0]]
    #
    # test = MLPClassifier(hidden_layer_widths=np.array([2]), momentum=0, deterministic=1, lr=1, shuffle=False, zero_weights=True)
    # test.fit(np.array(data), np.array(targ))

    # df = pd.DataFrame(data, index=[0])
    # df.plot(kind='bar')
    # plt.show()




if __name__ == "__main__":
    main()
    exit()
