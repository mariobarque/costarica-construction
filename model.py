import pandas as pd
import numpy as np


def create_k_trains(k, train_dataset):
    """
    Splits the train data set into K data sets
    Calculate the size of any data set using K / length of test dataset
    Loop through the K elements and calculate the indexes where to
    partition them and saved them into an array
    :param k: the number of data sets
    :param train_dataset: the dataset for training
    :return: the array of k training datasets
    """
    k_trains = []
    k_len = int(len(train_dataset)/k)
    for i in range(0, k):
        current_cutoff = i * k_len
        next_cutoff = current_cutoff + k_len

        if i + 1 == k:
            k_trains.append(train_dataset[current_cutoff:])
        else:
            k_trains.append(train_dataset[current_cutoff:next_cutoff])

    return k_trains



def test(df, weights, variable):
    """
    Giving a test data set and the wights, run the prediction (inference)
    perform Ax = b. Where A is the dataset, x are the weights, and
    b is the real value.
    :param df: the data frame with training dataset
    :param weights: the weights (x) value gotten out of models
    :param variable: the prediction variable
    :return: mean square error and prediction
    """
    b = df[variable].values
    current_test = df.drop(variable, 1)
    A = current_test.values

    prediction = np.dot(A, weights)
    error = prediction_error(prediction, b)

    return error, prediction


def test_network(df, forward_pass_test, wo, ws, variable):
    """
    For a multilayer perceptron inference is different.
    Instead of Ax=b, here we need to use weights from input layer ->
    hidden layer (wo) and weights from hidden layer -> output layer (ws)
    to feed forward the neural network and get the output.
    :param df: the data frame with test dataset
    :param forward_pass_test: the method pass forward in the network
    :param wo: weights from input layer -> hidden layer
    :param ws: weights from hidden layer -> output layer
    :param variable: the prediction variable
    :return: mean square error and output
    """
    b = df[variable].values
    current_test = df.drop(variable, 1)
    A = current_test.values

    prediction = forward_pass_test(wo, ws, A)
    error = prediction_error(prediction, b)

    return error, prediction


def prediction_error(prediction, b):
    """
    Mean square error
    :param prediction: the prediction
    :param b: the real value
    :return: the mean square error between prediction and real value
    """
    return np.sqrt(np.sum((prediction - b) ** 2))


def train(train_dataset, k, model, prediction_variable):
    """
    Performs k-fold cross train validation. Split dataset into K datasets
    train all of them minus one and test it with the one that wasn't trained with.
    Repeat the process K times and average weights at the end.
    For example. Dataset: A1, A2, A3, A4
        K1 = A1: train = A2, A3, A4, test = A1
        K1 = A2: train = A1, A3, A4, test = A2
        K1 = A3: train = A1, A2, A4, test = A3
        K1 = A4: train = A1, A2, A3, test = A4

    :param train_dataset: the training dataset
    :param k: how many data set to split testing data for cross validation
    :param model: the function that represents the model used for training
    :param prediction_variable: the prediction variable
    :return: the average weights (x) and the errors
    """
    k_trains = create_k_trains(k, train_dataset)
    weights_list = []
    errors = []
    for i in range(0, k):
        current_train = pd.DataFrame()
        for j in range(0, k):
            if j != i:
                current_train = current_train.append(k_trains[j], ignore_index=True)

        current_test = k_trains[i]
        weights = model(current_train, prediction_variable)
        error, prediction = test(current_test, weights, prediction_variable)

        weights_list.append(weights)
        errors.append(error)

    avg_weights = [np.mean(a) for a in zip(*weights_list)]

    return avg_weights, errors


'''
@train_dataset 
@k 
@model 
@prediction_variable 
@epochs 
@alpha 
    Since a network doesn't just store a weight per each column but instead it stores a weight
    per each node in the graph then training it using k-fold is different.
    Per every K iteration store wo and ws and the error. 
    At the end perform mean for the error, wo and ws and returns all errors.
Return the errors, average weights from input layer -> hidden layer and 
average weights from hidden layer -> output layer  
'''
def train_network(train_dataset, k, model, prediction_variable, hidden_layer_size, epochs = 50, alpha = 0.1):
    """

    :param train_dataset: the training dataset
    :param k: how many data set to split testing data for cross validation
    :param model: the function that represents the model used for training
    :param prediction_variable: the prediction variable
    :param hidden_layer_size: the hidden layer size
    :param epochs: the number of iterations we want to run adjust weights and feed forward in the neural network
    :param alpha: how fast to adjust the weights
    :return:
    """
    k_trains = create_k_trains(k, train_dataset)
    errors_list = []
    wo_list = []
    ws_list = []

    for i in range(0, k):
        current_train = pd.DataFrame()
        for j in range(0, k):
            if j != i:
                current_train = current_train.append(k_trains[j], ignore_index=True)

        current_test = k_trains[i]
        errors, wo, ws = model(current_test, hidden_layer_size, prediction_variable, epochs, alpha)
        errors_list.append(errors)
        wo_list.append(wo.numpy())
        ws_list.append(ws.numpy())

    avg_wo = np.mean(np.array(wo_list), axis=0)
    avg_ws = np.mean(np.array(ws_list), axis=0)
    avg_errors = np.mean(np.array(errors_list), axis=0)

    return avg_errors, avg_wo, avg_ws