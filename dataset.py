import pandas as pd
import numpy as np

all_columns = ['anoper', 'num_obras', 'arecon', 'valobr', 'claper', 'claobr', 'numpis', 'numviv', 'numapo', 'numdor',
               'matpis', 'matpar', 'mattec', 'usoobr', 'financ', 'cod_provincia', 'provincia', 'id_canton', 'canton',
               'id_region', 'cat']

numeric_columns = ['num_obras', 'arecon', 'numpis', 'numviv', 'numapo', 'numdor']

categorical_columns = ['anoper', 'claper', 'claobr', 'matpis', 'matpar', 'mattec', 'usoobr', 'financ',
                       'cod_provincia', 'id_region', 'cat']

extra_categorical_columns = ['num_obras_cat', 'arecon_cat', 'numpis_cat', 'numviv_cat', 'numapo_cat', 'numdor_cat']

data = []


def load_data(path):
    global data
    data = pd.read_csv(path)
    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    numerical_to_categorical(data)


def get_data_set():
    columns = numeric_columns + categorical_columns
    df = data[columns]
    return df


def get_data_set_for_analysis():
    columns = categorical_columns + extra_categorical_columns
    df = data[columns]
    return df


def get_numerical_data_set():
    df = data[numeric_columns]
    return df


def get_categorical_data_set():
    df = data[categorical_columns]
    return df


def numerical_to_categorical(df):
    df['num_obras_cat'] = pd.cut(df['num_obras'], [-1, 0, 1, 4, 10, 20, np.inf],
                                 labels=['0', '1', '4', '10', '20', '20+'])

    df['arecon_cat'] = pd.cut(df['arecon'], [-1, 0, 50, 100, 1000, np.inf],
                              labels=['0', '50', '100', '1000', '1000+'])

    df['numpis_cat'] = pd.cut(df['numpis'], [-1, 0, 1, np.inf],
                              labels=['0', '1', '1+'])

    df['numviv_cat'] = pd.cut(df['numviv'], [-1, 0, 1, np.inf],
                              labels=['1', '2', '3'])

    df['numapo_cat'] = pd.cut(df['numapo'], [-1, 0, 4, 10, np.inf],
                              labels=['1', '2', '3', '4'])

    df['numdor_cat'] = pd.cut(df['numdor'], [-1, 0, 4, 10, np.inf],
                              labels=['1', '2', '3', '4'])
