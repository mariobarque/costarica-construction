import matplotlib.pyplot as plt;
import seaborn as sns
sns.set(color_codes=True)
plt.rcParams["figure.figsize"] = (18, 18)


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


def plot_distributions(df):
    fig, axs = plt.subplots(ncols=4, nrows=3, squeeze=False)
    sns.distplot(df['numpis'], ax=axs[0, 0])
    sns.distplot(df['numviv'], ax=axs[0, 1])
    sns.distplot(df['numapo'], ax=axs[0, 2])
    sns.distplot(df['numdor'], ax=axs[0, 3])
    sns.distplot(df['cod_provincia'], ax=axs[1, 0])
    sns.distplot(df['id_region'], ax=axs[1, 1])
    sns.distplot(df['valobr'], ax=axs[1, 2])
    sns.distplot(df['arecon'], ax=axs[1, 3])

    sns.distplot(df['matpis'], ax=axs[2, 2])
    sns.distplot(df['matpar'], ax=axs[2, 3])
    sns.distplot(df['mattec'], ax=axs[2, 0])
    sns.distplot(df['usoobr'], ax=axs[2, 1])