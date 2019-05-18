import matplotlib.pyplot as plt
import seaborn as sns


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

