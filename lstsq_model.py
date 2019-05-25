import numpy as np

def train(df, variable):
    b = df[variable].values
    current_train = df.drop(variable, 1)
    A = current_train.values
    weights = np.linalg.lstsq(A, b)[0]

    return weights