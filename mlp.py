import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import copy



class MLPClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, hidden_layer_width=None, lr=.1, momentum=0, shuffle=True, deterministic=None,
                 use_zero_weights=False, valid_size=0.):
        self.hidden_layer_width = hidden_layer_width
        self.lr = lr            # Learning rate. It's how much you update the weights. higher = faster but less accurate.
        self.momentum = momentum
        self.shuffle = shuffle
        self.weights_input_to_hidden = None
        self.weights_hidden_to_output = None
        self.momentum_first_weights = None
        self.momentum_second_weights = None
        self.use_zero_weights = use_zero_weights
        self.valid_size = valid_size
        if deterministic is not None:
            self.deterministic = True
            self.determine_epochs = deterministic
        else:
            self.deterministic = False

    def fit(self, X, y, initial_weights=None):
        """ 
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        if self.shuffle:
            X, y = self._shuffle_data(X, y)

        validation_set, validation_targets, test_set, test_targets, X, y = self._get_valid_test_sets(X, y)

        self.initialize_weights(X, y)

        num_epochs = 0
        min_loss = np.inf
        times_unchanged = 0

        accuracies = []
        MSE_valid = []
        MSE_train = []
        MSE_test = []

        if self.deterministic:
            num_epochs = self.determine_epochs
            for x in range(self.determine_epochs):
                self.do_epoch(X, y)

                mse, acc = self.score(validation_set, validation_targets)
                mse_test, _ = self.score(test_set, test_targets)
                mse_train, _ = self.score(X, y)

                accuracies.append(acc)
                MSE_valid.append(mse)
                MSE_train.append(mse_train)
                MSE_test.append(mse_test)
        else:
            while times_unchanged < 10:
                num_epochs += 1
                self.do_epoch(X, y)
                mse_valid, acc = self.score(validation_set, validation_targets)
                mse_train, _ = self.score(X, y)
                mse_test, _ = self.score(test_set, test_targets)

                accuracies.append(acc)
                MSE_valid.append(mse_valid)
                MSE_train.append(mse_train)
                MSE_test.append(mse_test)

                if mse_valid < min_loss:
                    min_loss = mse_valid
                    times_unchanged = 0
                else:
                    times_unchanged += 1
        print(f"Num epochs = {num_epochs}")
        print(f"MSE = {self.score(validation_set, validation_targets)}")

        return MSE_valid, accuracies, MSE_train, num_epochs, MSE_test

    def _get_valid_test_sets(self, data, targets):
        num_data = data.shape[0]        # Rows in data set

        if self.valid_size != 0:
            valid_num = num_data * self.valid_size // 1
            validation_index = num_data - valid_num
            validation_index = int(validation_index)
            validation_set = data[validation_index:, :]
            validation_targets = targets[validation_index:, :]
            data = data[:validation_index, :]
            targets = targets[:validation_index, :]

            num_data = data.shape[0]
            test_index = num_data - valid_num
            test_index = int(test_index)
            test_set = data[test_index:, :]
            test_targets = targets[test_index:, :]
            data = data[:test_index, :]
            targets = targets[:test_index, :]

        else:
            validation_set = data
            validation_targets = targets
            test_set = data
            test_targets = targets

        return validation_set, validation_targets, test_set, test_targets, data, targets

    def do_epoch(self, data, targets):
        if self.shuffle:
            data, targets = self._shuffle_data(data.copy(), targets.copy())

        for (row, targ) in zip(data, targets):
            prediction, hidden_outs = self._predict_one(row)
            row = np.append(row, 1)

            #    error = {targ - prediction}; {prediction * (1 - prediction)} = the derivative of the sigmoid function
            error_pred = (targ - prediction) * prediction * (1 - prediction)
            # Doing this error * derivative of sigmoid sets us up to do back prop.

            error_hidden = np.matmul(error_pred, self.weights_hidden_to_output.T)
            error_hidden *= (1 - hidden_outs) * hidden_outs      # This (1-out)*out = derivative of sigmoid func
            error_hidden = error_hidden[:-1]                     # Remove bias, necessary to compute delta input weights

            # delta_first weights = row.T * error_hidden
            delta_first_weights = np.matmul(row.reshape((len(row), 1)), error_hidden.reshape((1, len(error_hidden))))
            # delta_second_weights = hidden_outs.T * error_pred
            delta_second_weights = np.matmul(hidden_outs.reshape((len(hidden_outs), 1)), error_pred.reshape((1, len(error_pred))))

            delta_first_weights *= self.lr
            delta_second_weights *= self.lr
            delta_first_weights += self.momentum * self.momentum_first_weights
            delta_second_weights += self.momentum * self.momentum_second_weights

            self.momentum_first_weights = delta_first_weights.copy()
            self.momentum_second_weights = delta_second_weights.copy()
            self.weights_input_to_hidden += delta_first_weights                     # Update the weights with every data
            self.weights_hidden_to_output += delta_second_weights                   # instance

    def _predict_one(self, input_instance):
        input_instance = np.append(input_instance.copy(), 1)       # This is to add a bias to the input.

        hidden_net = np.matmul(input_instance, self.weights_input_to_hidden)
        hidden_output = self._sigmoid(hidden_net)
        hidden_output = np.append(hidden_output, 1)     # This is to add a bias to the hidden layer output

        prediction_net = np.matmul(hidden_output, self.weights_hidden_to_output)
        prediction = self._sigmoid(prediction_net)
        return prediction, hidden_output

    def predict(self, data):
        """ Predict all classes for a dataset data
        Args:
            data (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in data.
        """
        guesses = []
        data = copy.copy(data)
        for row in data:
            prediction, _ = self._predict_one(row)
            guesses.append(np.argmax(prediction))
        return guesses

    def initialize_weights(self, data, targets):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        if self.hidden_layer_width is None:
            self.hidden_layer_width = [data.shape[1] * 2]

        hidden_layer_size = self.hidden_layer_width[0]

        num_inputs = data.shape[1]
        self.momentum_first_weights = np.zeros((num_inputs + 1, hidden_layer_size))

        if self.use_zero_weights:
            first_weights = np.zeros((num_inputs + 1, hidden_layer_size))
        else:
            first_weights = np.random.normal(loc=0, scale=1, size=(num_inputs + 1, hidden_layer_size))
        num_outputs = targets.shape[1]
        if self.use_zero_weights:
            second_weights = np.zeros((hidden_layer_size + 1, num_outputs))
        else:
            second_weights = np.random.normal(loc=0, scale=1, size=(hidden_layer_size + 1, num_outputs))

        self.momentum_second_weights = np.zeros((hidden_layer_size + 1, num_outputs))
        self.weights_input_to_hidden = first_weights
        self.weights_hidden_to_output = second_weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        if self.weights_input_to_hidden is None or self.weights_hidden_to_output is None:
            self.initialize_weights(X, y)
        num_right = 0
        total = 0
        MSE = 0
        for row, targ in zip(X, y):
            row = np.append(row, 1)
            hidden_outs = np.matmul(row, self.weights_input_to_hidden)
            hidden_outs = self._sigmoid(hidden_outs)
            hidden_outs = np.append(hidden_outs, 1)
            output = np.matmul(hidden_outs, self.weights_hidden_to_output)
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
        """ Shuffle the data! 
        """
        y_size = y.shape[1]
        X = np.hstack((X, y))
        if self.shuffle:
            np.random.shuffle(X)
        return X[:, :-y_size], X[:, -y_size:]

    @staticmethod
    def _sigmoid(output):
        return np.reciprocal(1 + np.exp(output * -1))

    def get_weights(self):
        return self.weights_input_to_hidden, self.weights_hidden_to_output
    #
