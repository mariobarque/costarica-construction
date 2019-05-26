import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


'''
@df the data frame
@val the threshold
    Creates a problem of regression into classification by converting the prediction variable
    into a value that can be only 0 or 1
Returns the data set with an extra binary column
'''
def set_prediction_variable(df, val):
    df['cat'] = df['valobr'] > val
    return df

'''
@data the dataframe
    Based on the information from Wikipedia get the region from the province and canton 
Returns the region code
'''
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


'''
@df the dataframe with the dat
@cols the columns array
    Plot the distribution for all the numeric columns in the cols array
'''
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


'''
@df the dataframe with the data
@cols the columns array
    Plot the distribution for all the columns in the cols array
'''
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


'''
@k_trains the k training datasets
    Plot the distribution of the category (variable to predict) in
    every k dataset
'''
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


'''
@real_value the real value
@prediction the prediction
    Plot the Receiver Operating Characteristic (ROC) curve
'''
def plot_roc_curve(real_value, prediction, error):
    auc = roc_auc_score(real_value, prediction)
    print('AUC: ', auc)
    print('Error: ', error)

    fpr, tpr, thresholds = roc_curve(real_value, prediction)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Falsos Negativos')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


'''
@errors an array of error values
    Plot the error value in every element in the array creating a curve
'''
def plot_error(errors):
    axes = plt.subplots()[1]
    axes.plot(errors, color='red', label='error')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.title('Error')
    plt.show()