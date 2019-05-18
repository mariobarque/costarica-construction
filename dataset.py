import pandas as pd
import utilities

all_columns = ['anoper', 'num_obras', 'arecon', 'valobr', 'claper', 'claobr', 'numpis', 'numviv', 'numapo', 'numdor',
               'matpis', 'matpar', 'mattec', 'usoobr', 'financ', 'cod_provincia', 'provincia', 'id_canton', 'canton',
               'id_region', 'cat']

numeric_columns = ['num_obras', 'arecon', 'numpis', 'numviv', 'numapo', 'numdor']
categorical_columns = ['anoper', 'claper', 'claobr', 'matpis', 'matpar', 'mattec', 'usoobr', 'financ',
                       'cod_provincia', 'id_canton', 'id_region', 'cat']
extra_categorical_columns = ['num_obras_cat', 'arecon_cat', 'numpis_cat', 'numviv_cat', 'numapo_cat', 'numdor_cat']

data = []


def load_data(path):
    global data
    data = pd.read_csv(path)
    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    utilities.numerical_to_categorical(data)


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

