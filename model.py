import pandas as pd
import numpy as np


def create_k_trains(k, train):
    k_trains = []
    k_len = int(len(train)/k)
    for i in range(0, k):
        current_cutoff = i * k_len
        next_cutoff = current_cutoff + k_len

        if i + 1 == k:
            k_trains.append(train[current_cutoff:])
        else:
            k_trains.append(train[current_cutoff:next_cutoff])

    return k_trains

def test_model(df, weights, variable):
    b = df[variable].values
    current_test = df.drop(variable, 1)
    A = current_test.values

    prediction = np.dot(A, weights)
    prediction_error = np.sqrt(np.sum((prediction - b)**2))

    return prediction_error, prediction


def train_model(train_dataset, k, model, prediction_variable):
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
        error, prediction = test_model(current_test, weights, prediction_variable)

        weights_list.append(weights)
        errors.append(error)

    avg_weights = [np.mean(a) for a in zip(*weights_list)]

    return avg_weights, errors