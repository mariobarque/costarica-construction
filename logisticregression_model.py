import torch
import torch.nn as nn


def evaluate_sigmoid(W, X):
    """
    Perform the sigmoid function of the multiplication between weights and dataset
    :param W: weights
    :param X: the datasets
    :return: Returns the activation
    """
    net_weights = torch.mm(X, W)
    sigmoid = nn.Sigmoid()
    activation = sigmoid(net_weights)
    return activation


def get_delta_weights_for_class(weights, samples, alpha, target):
    """
    Runs activation function which is the sigmoid function.
    With the activation samples calculate the new error.
    With the new error calculate the delta weights
    And re calculate the weights based on alpha and the delta weights
    :param weights: the previous weights
    :param samples: the data set
    :param alpha: how fast to adjust the weights
    :param target: real values
    :return: the delta between target and prediction
    """
    activation_samples = evaluate_sigmoid(weights, samples)
    error = (target - activation_samples)
    delta_weights = samples * error
    delta_weights_with_alpha = alpha * delta_weights
    return delta_weights_with_alpha


def train_logistic_regression(class_zeros, class_ones, iterations, alpha):
    """
    Loop 'iterations' times. Every iterations
        - Get delta weights for class zero
        - Get delta weights for class one
        - Calculate the total weights
    :param class_zeros: the samples with the category value in zero
    :param class_ones: the samples with the category value in one
    :param iterations: number of iterations to run the regression
    :param alpha: how fast to adjust the weights
    :return: Returns the weights
    """
    class_zeros = torch.from_numpy(class_zeros).float()
    class_ones = torch.from_numpy(class_ones).float()

    dimensionsData = class_zeros.size()
    weights = torch.ones(dimensionsData[1],1)

    for i in range(0, iterations):
        delta_weights_zero = get_delta_weights_for_class(weights, class_zeros, alpha, target = 0)
        delta_weights_one = get_delta_weights_for_class(weights, class_ones, alpha, target = 1)
        delta_weight_total = torch.sum(delta_weights_zero, dim = 0) + torch.sum(delta_weights_one, dim = 0)
        weights = weights + delta_weight_total.view(len(weights), 1)

    return weights.numpy().T[0]


def train(df, variable, iterations = 50, alpha = 0.1):
    """
    Creates two classes, the one for ones and the one for zeros
    Make logistic regression for binary classification with these two clases
    :param df: the dataset
    :param variable: the prediction variable
    :param iterations: number of iterations to run the regression
    :param alpha: how fast to adjust the weights
    :return: the weights
    """
    classZero = df.loc[df[variable] == 0].drop(variable, 1)
    classOne = df.loc[df[variable] == 1].drop(variable, 1)

    weights = train_logistic_regression(classZero.values, classOne.values, iterations, alpha)

    return weights
