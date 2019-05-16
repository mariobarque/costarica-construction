import pandas as pd

all_columns = ['anoper', 'num_obras', 'arecon', 'valobr', 'claper', 'claobr', 'numpis', 'numviv', 'numapo', 'numdor',
               'matpis', 'matpar', 'mattec', 'usoobr', 'financ', 'cod_provincia', 'provincia', 'id_canton', 'canton',
               'id_region']

numeric_columns = ['num_obras', 'arecon', 'valobr', 'numpis', 'numviv', 'numapo', 'numdor']
categorical_columns = ['anoper', 'claper', 'claobr', 'matpis', 'matpar', 'mattec', 'usoobr', 'financ',
                       'cod_provincia', 'id_canton', 'id_region', 'cat']

data = []


def load_data(path, threshold):
    global data
    data = pd.read_csv(path)
    #data = set_prediction_variable(data, threshold)


def get_data_set():
    columns = numeric_columns + categorical_columns
    df = data[columns]
    return df


def get_numerical_data_set():
    df = data[numeric_columns]
    return df


def get_categorical_data_set():
    df = data[categorical_columns]
    return df


def set_prediction_variable(df, val):
    #df.loc[df['valobr'] > val, 'cat'] = 1
    #df.loc[df['valobr'] <= val, 'cat'] = 0
    df['cat'] = 1 if df['valobr'] > val else 0
    return df
