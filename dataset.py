import pandas as pd

all_columns = ['anoper', 'num_obras', 'arecon', 'valobr', 'claper', 'claobr', 'numpis', 'numviv', 'numapo', 'numdor',
               'matpis', 'matpar', 'mattec', 'usoobr', 'financ', 'cod_provincia', 'provincia', 'id_canton', 'canton',
               'id_region']

numeric_columns = ['num_obras', 'arecon', 'valobr', 'numpis', 'numviv', 'numapo', 'numdor']
categorical_columns = ['anoper', 'claper', 'claobr', 'matpis', 'matpar', 'mattec', 'usoobr', 'financ',
                       'cod_provincia', 'id_canton', 'id_region']

data = []


def load_data():
    global data
    data = pd.read_csv('data/construction-data-processed.csv')


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

