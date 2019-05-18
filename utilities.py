import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



def set_prediction_variable(df, val):
    df['cat'] = df['valobr'] > val
    return df


def get_region(data):
    # Chorotega: all Guanacaste plus Upala in Alajuela
    if data['cod_provincia'] == 5 or data['id_canton'] == 33:
        return 2

    # Pacífico Central: Some cantones from Puntarenas and some from Alajuela
    if data['id_canton'] in [24, 29, 65, 66, 68, 70, 73, 75]:
        return 3

    # Brunca: Some cantones from Puntarenas and Pérez Zeledón
    if data['id_canton'] in [19, 67, 69, 71, 72, 74]:
        return 4

    # Huetar Atlantica: All cantones from Limón
    if data['cod_provincia'] == 7:
        return 5

        # Huetar Norte: Some cantones from Alajuela
    if data['id_canton'] in [34, 35, 30]:
        return 6

    # The other cantones in Central Valley
    return 1


def plot_distributions(df, numeric_columns):
    if len(numeric_columns) != 6:
        return
    fig, axs = plt.subplots(ncols=3, nrows=2, squeeze=False)

    sns.distplot(df[numeric_columns[0]], ax=axs[0, 0])
    sns.distplot(df[numeric_columns[1]], ax=axs[0, 1])
    sns.distplot(df[numeric_columns[2]], ax=axs[0, 2])
    sns.distplot(df[numeric_columns[3]], ax=axs[1, 0])
    sns.distplot(df[numeric_columns[4]], ax=axs[1, 1])
    sns.distplot(df[numeric_columns[5]], ax=axs[1, 2])


def plot_categorical_distributions(df, cols):
    if len(cols) != 16:
        return
    fig, axs = plt.subplots(ncols=3, nrows=6, squeeze=False)

    sns.countplot(x=cols[0], data=df, ax=axs[0, 0])
    sns.countplot(x=cols[1], data=df, ax=axs[0, 1])
    sns.countplot(x=cols[2], data=df, ax=axs[0, 2])
    sns.countplot(x=cols[3], data=df, ax=axs[1, 0])
    sns.countplot(x=cols[4], data=df, ax=axs[1, 1])
    sns.countplot(x=cols[5], data=df, ax=axs[1, 2])
    sns.countplot(x=cols[6], data=df, ax=axs[2, 0])
    sns.countplot(x=cols[7], data=df, ax=axs[2, 1])
    sns.countplot(x=cols[8], data=df, ax=axs[2, 2])
    sns.countplot(x=cols[9], data=df, ax=axs[3, 0])
    sns.countplot(x=cols[10], data=df, ax=axs[3, 1])
    sns.countplot(x=cols[11], data=df, ax=axs[3, 2])
    sns.countplot(x=cols[12], data=df, ax=axs[4, 0])
    sns.countplot(x=cols[13], data=df, ax=axs[4, 1])
    sns.countplot(x=cols[14], data=df, ax=axs[4, 2])
    sns.countplot(x=cols[15], data=df, ax=axs[5, 0])


def plot_k_folds(k_trains):
    fig, axs = plt.subplots(ncols=3, nrows=3, squeeze=False)
    sns.countplot(x='cat', data=k_trains[0], ax=axs[0, 0])
    sns.countplot(x='cat', data=k_trains[1], ax=axs[0, 1])
    sns.countplot(x='cat', data=k_trains[2], ax=axs[0, 2])
    sns.countplot(x='cat', data=k_trains[3], ax=axs[1, 0])
    sns.countplot(x='cat', data=k_trains[4], ax=axs[1, 1])
    sns.countplot(x='cat', data=k_trains[5], ax=axs[1, 2])
    sns.countplot(x='cat', data=k_trains[6], ax=axs[2, 0])
    sns.countplot(x='cat', data=k_trains[7], ax=axs[2, 1])
    sns.countplot(x='cat', data=k_trains[8], ax=axs[2, 2])


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


def train_model_lstsq(df, variable):
    b = df['cat'].values
    current_train = df.drop(variable, 1)
    A = current_train.values
    weights = np.linalg.lstsq(A, b)[0]

    return weights


def test_model(df, weights, variable):
    b = df[variable].values
    current_test = df.drop(variable, 1)
    A = current_test.values

    prediction = np.dot(A, weights)
    #prediction = np.array([np.abs(round(x)) for x in prediction]).astype(int)
    prediction_error = np.sqrt(np.sum((prediction - b)**2))

    return prediction_error, prediction


def k_fold_train_model(k_trains, k, variable):
    weights_list = []
    errors = []
    for i in range(0, k):
        current_train = pd.DataFrame()
        for j in range(0, k):
            if j != i:
                current_train = current_train.append(k_trains[j], ignore_index=True)

        current_test = k_trains[i]
        weights = train_model_lstsq(current_train, variable)
        error, prediction = test_model(current_test, weights, variable)

        weights_list.append(weights)
        errors.append(error)

    avg_weights = [np.mean(a) for a in zip(*weights_list)]

    return avg_weights, errors


def plot_roc_curve(prediction, real_value):
    fpr, tpr, thresholds = roc_curve(real_value, prediction)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Falsos Negativos')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

