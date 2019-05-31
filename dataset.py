import pandas as pd
import numpy as np

all_columns = ['anoper', 'num_obras', 'arecon', 'valobr', 'claper', 'claobr', 'numpis', 'numviv', 'numapo', 'numdor',
               'matpis', 'matpar', 'mattec', 'usoobr', 'financ', 'cod_provincia', 'provincia', 'id_canton', 'canton',
               'id_region', 'cat']

numeric_columns = ['num_obras', 'arecon', 'numpis', 'numviv', 'numapo', 'numdor']

categorical_columns = ['anoper', 'claper', 'claobr', 'matpis', 'matpar', 'mattec', 'usoobr', 'financ',
                       'cod_provincia', 'id_region']

extra_categorical_columns = ['num_obras_cat', 'arecon_cat', 'numpis_cat', 'numviv_cat', 'numapo_cat', 'numdor_cat']

data = []


'''
@path the file's path inside the repo
    Load the data from the csv into a pandas dataframe
    Transform the numerical values into categorical values
'''
def load_data(path):
    global data
    data = pd.read_csv(path)
    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    numerical_to_categorical(data)


'''
    In case someone wants the numerical + categorical values
Returns the dataframe with numerical an categorical columns
'''
def get_data_set():
    columns = numeric_columns + categorical_columns
    df = data[columns]
    return df


'''
@path the csv file's path
@N the number of samples to limit. If not set, then it'll used all possible values
@balanced whether we want the data balanced or not
    - Load data, since all data is set to categorical data all columns will be full of
      ones an zeros.
    - Add the prediction variable. Since we need a balanced dataset we need to find the
      size of category with less elements and that quantity of samples from both categories. 
Returns the dataframe ready to be run in models.
'''
def get_data_for_model(path, N = None, balanced = True):
    load_data(path)
    cols = categorical_columns + extra_categorical_columns

    df = get_data_set_for_analysis()
    df_encoded = pd.get_dummies(data=df, columns=cols)

    # add the prediction variable
    df_encoded['cat'] = df['cat']

    if not balanced:
        return df_encoded.sample(frac=1).reset_index(drop=True)

    # Balance the data
    total_class_zero = df_encoded['cat'].where(lambda cat: cat == 0).count()
    total_class_one = df_encoded['cat'].where(lambda cat: cat == 1).count()
    min_class_size = np.min([total_class_zero, total_class_one])

    if N is None:
        N = min_class_size * 2
    else:
        if N > (min_class_size * 2):
            N = min_class_size * 2

    # Obtenemos los datos y unimos las clases
    class_zero = df_encoded.loc[df['cat'] == 0].head(int(N/2))
    class_one = df_encoded.loc[df['cat'] == 1].head(int(N/2))
    df_analysis = pd.concat([class_zero, class_one], ignore_index=True)

    # Revolvemos los datos aleatóriamente
    df_analysis = df_analysis.sample(frac=1).reset_index(drop=True)

    return df_analysis


'''
Returns the dataframe with all categorical values already converted by one hot encoding
'''
def get_data_set_for_analysis():
    columns = categorical_columns + extra_categorical_columns
    columns.append('cat') # variable to predict
    df = data[columns]
    return df


'''
Returns the dataframe with all numerical values
'''
def get_numerical_data_set():
    df = data[numeric_columns]
    return df


'''
Returns the dataframe with all categorical values 
'''
def get_categorical_data_set():
    df = data[categorical_columns]
    return df


'''
@df The dataframe
    Convert the original numerical values into categories.
'''
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
