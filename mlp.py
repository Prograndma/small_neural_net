import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import copy


### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.


class MLPClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, hidden_layer_widths=None, lr=.1, momentum=0, shuffle=True, deterministic=None, zero_weights=False, valid_size=0.):
        """ Initialize class with chosen hyperparameters.
        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.first_weights = None
        self.second_weights = None
        self.momentum_first_weights = None
        self.momentum_second_weights = None
        self.zero_weights = zero_weights
        self.valied_size = valid_size
        if deterministic is not None:
            self.deterministic = True
            self.determine_epochs = deterministic
        else:
            self.deterministic = False

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        num_data = X.shape[0]
        if self.shuffle:
            X, y = self._shuffle_data(X, y)
        # TODO: test, validation and training sets 30\30\40, 25\25\50
        if self.valied_size != 0:
            valid_num = num_data * self.valied_size // 1
            validation_index = num_data - valid_num
            validation_index = int(validation_index)
            validation_set = X[validation_index:, :]
            validation_targets = y[validation_index:, :]
            X = X[:validation_index, :]
            y = y[:validation_index, :]

            num_data = X.shape[0]
            test_index = num_data - valid_num
            test_index = int(test_index)
            test_set = X[test_index:, :]
            test_targets = y[test_index:, :]
            X = X[:test_index, :]
            y = y[:test_index, :]

        else:
            validation_set = X
            validation_targets = y
            test_set = X
            test_targets = y
        # old_x = copy.copy(X)
        self.initialize_weights(X, y)
        num_epochs = 0
        min_loss = np.inf
        times_unchanged = 0
        losses = []
        MSE_train = []
        accuracies = []
        MSE_test = []
        if self.deterministic:
            num_epochs = self.determine_epochs
            for x in range(self.determine_epochs):
                if self.shuffle:
                    to_x, to_y = self._shuffle_data(X, y)
                    self.do_epoch(to_x, to_y)
                else:
                    self.do_epoch(X, y)
                mse, acc = self.score(validation_set, validation_targets)
                accuracies.append(acc)
                losses.append(mse)
                mse, acc = self.score(X, y)
                MSE_train.append(mse)
                mse, acc = self.score(test_set, test_targets)
                MSE_test.append(mse)

                print(f"Accuracy = [{acc}]")

        else:
            while times_unchanged < 10:
                # loss, acc = self.score(validation_set, validation_targets)
                # losses.append(loss)
                # accuracies.append(acc)
                num_epochs += 1
                if self.shuffle:
                    to_x, to_y = self._shuffle_data(X, y)
                    self.do_epoch(to_x, to_y)
                else:
                    self.do_epoch(X, y)
                temp_loss, acc = self.score(validation_set, validation_targets)
                losses.append(temp_loss)
                accuracies.append(acc)
                mse, acc = self.score(X, y)
                MSE_train.append(mse)
                mse, acc = self.score(test_set, test_targets)
                MSE_test.append(mse)
                if temp_loss < min_loss:
                    min_loss = temp_loss
                    times_unchanged = 0
                else:
                    times_unchanged += 1
        print(f"Num epochs = {num_epochs}")
        print(f"MSE = {self.score(validation_set, validation_targets)}")

        return losses, accuracies, MSE_train, num_epochs, MSE_test

    def do_epoch(self, data, targets):
        for (row, targ) in zip(data, targets):
            row = np.append(row, 1)
            hidden_outs = np.matmul(row, self.first_weights)
            # DO this??
            hidden_outs = self._sigmoid(hidden_outs)
            hidden_outs = np.append(hidden_outs, 1)
            prediction = np.matmul(hidden_outs, self.second_weights)
            prediction = self._sigmoid(prediction)
            error_pred = (targ - prediction) * prediction * (1 - prediction)
            # Something is going on here.
            error_hidden = np.matmul(error_pred, self.second_weights.T)
            error_hidden *= (1 - hidden_outs) * hidden_outs
            error_hidden = error_hidden[:-1]
            delta_first_weights = np.matmul(row.reshape((len(row), 1)), error_hidden.reshape((1, len(error_hidden))))
            delta_second_weights = np.matmul(hidden_outs.reshape((len(hidden_outs), 1)), error_pred.reshape((1, len(error_pred))))
            delta_first_weights *= self.lr
            delta_second_weights *= self.lr
            delta_first_weights += self.momentum * self.momentum_first_weights
            delta_second_weights += self.momentum * self.momentum_second_weights
            self.momentum_first_weights = delta_first_weights.copy()
            self.momentum_second_weights = delta_second_weights.copy()
            self.first_weights += delta_first_weights
            self.second_weights += delta_second_weights

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        guesses = []
        X = copy.copy(X)
        for row in X:
            row = np.append(row, 1)
            hidden_outs = np.matmul(row, self.first_weights)
            hidden_outs = self._sigmoid(hidden_outs)
            hidden_outs = np.append(hidden_outs, 1)
            prediction = np.matmul(hidden_outs, self.second_weights)
            prediction = self._sigmoid(prediction)
            guesses.append(np.argmax(prediction))
        return guesses

    def initialize_weights(self, data, targets):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        if self.hidden_layer_widths is None:
            self.hidden_layer_widths = [data.shape[1] * 2]

        hidden_layer_size = self.hidden_layer_widths[0]

        num_inputs = data.shape[1]
        self.momentum_first_weights = np.zeros((num_inputs + 1, hidden_layer_size))

        if self.zero_weights:
            first_weights = np.zeros((num_inputs + 1, hidden_layer_size))
        else:
            first_weights = np.random.normal(loc=0, scale=1, size=(num_inputs + 1, hidden_layer_size))
        num_outputs = targets.shape[1]
        if self.zero_weights:
            second_weights = np.zeros((hidden_layer_size + 1, num_outputs))
        else:
            second_weights = np.random.normal(loc=0, scale=1, size=(hidden_layer_size + 1, num_outputs))

        self.momentum_second_weights = np.zeros((hidden_layer_size + 1, num_outputs))
        self.first_weights = first_weights
        self.second_weights = second_weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        if self.first_weights is None or self.second_weights is None:
            self.initialize_weights(X, y)
        num_right = 0
        total = 0
        MSE = 0
        for row, targ in zip(X, y):
            row = np.append(row, 1)
            hidden_outs = np.matmul(row, self.first_weights)
            hidden_outs = self._sigmoid(hidden_outs)
            hidden_outs = np.append(hidden_outs, 1)
            output = np.matmul(hidden_outs, self.second_weights)
            output = self._sigmoid(output)
            MSE += (np.sum((targ - output) ** 2))
            max = np.argmax(output)
            # if max != 1:
            #     print('Wow')
            if np.argmax(output) == np.argmax(targ):
                num_right += 1
            total += 1
        MSE /= X.shape[0]
        return MSE, (num_right / total)

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        y_size = y.shape[1]
        X = np.hstack((X, y))
        if self.shuffle:
            np.random.shuffle(X)
        return X[:, :-y_size], X[:, -y_size:]

    @staticmethod
    def _sigmoid(output):
        return np.reciprocal(1 + np.exp(output * -1))

    # Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.first_weights, self.second_weights

    @staticmethod
    def _thresh_hold(num):
        pass
