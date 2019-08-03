import numpy as np


def train(df, variable):
    """
    Perform least square calculation
    A is the dataset with all columns
    b is the prediction variable column
    :param df: the data frame with training dataset
    :param variable: the prediction variable
    :return: the weights from least square
    """
    b = df[variable].values
    current_train = df.drop(variable, 1)
    A = current_train.values
    weights = np.linalg.lstsq(A, b, rcond=None)[0]

    return weights