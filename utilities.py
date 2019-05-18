import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def numerical_to_categorical(df):
    df['num_obras_cat'] = pd.cut(df['num_obras'], [-1, 0, 1, 4, 10, 20, np.inf],
                                 labels=['1', '2', '3', '4', '5', '6'])

    df['arecon_cat'] = pd.cut(df['arecon'], [-1, 0, 50, 100, 1000, np.inf],
                              labels=['1', '2', '3', '4', '5'])

    df['numpis_cat'] = pd.cut(df['numpis'], [-1, 0, 1, np.inf],
                              labels=['1', '2', '3'])

    df['numviv_cat'] = pd.cut(df['numviv'], [-1, 0, 1, np.inf],
                              labels=['1', '2', '3'])

    df['numapo_cat'] = pd.cut(df['numapo'], [-1, 0, 4, 10, np.inf],
                              labels=['1', '2', '3', '4'])

    df['numdor_cat'] = pd.cut(df['numdor'], [-1, 0, 4, 10, np.inf],
                              labels=['1', '2', '3', '4'])
