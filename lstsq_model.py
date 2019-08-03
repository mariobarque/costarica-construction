import numpy as np


'''
@df the data frame with training dataset
@variable the prediction variable
    Perform least square calculation 
    A is the dataset with all columns
    b is the prediction variable column
Return the weights from least square
'''
def train(df, variable):
    b = df[variable].values
    current_train = df.drop(variable, 1)
    A = current_train.values
    weights = np.linalg.lstsq(A, b, rcond=None)[0]

    return weights