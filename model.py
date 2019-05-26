import pandas as pd
import numpy as np


def create_k_trains(k, train_dataset):
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
    b = df[variable].values
    current_test = df.drop(variable, 1)
    A = current_test.values

    prediction = np.dot(A, weights)
    error = prediction_error(prediction, b)

    return error, prediction


def test_network(df, forward_pass_test, wo, ws, variable):
    b = df[variable].values
    current_test = df.drop(variable, 1)
    A = current_test.values

    prediction = forward_pass_test(wo, ws, A)
    error = prediction_error(prediction, b)

    return error, prediction


def prediction_error(prediction, b):
    return np.sqrt(np.sum((prediction - b) ** 2))


def train(train_dataset, k, model, prediction_variable):
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


def train_network(train_dataset, k, model, prediction_variable, epochs = 50, alpha = 0.1):
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
        errors, wo, ws = model(current_test, prediction_variable, epochs, alpha)
        errors_list.append(errors)
        wo_list.append(wo.numpy())
        ws_list.append(ws.numpy())

    avg_wo = np.mean(np.array(wo_list), axis=0)
    avg_ws = np.mean(np.array(ws_list), axis=0)

    return errors, avg_wo, avg_ws