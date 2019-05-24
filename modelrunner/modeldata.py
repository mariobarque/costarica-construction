import dataset
import numpy as np
import pandas as pd

def get_data(path):
    dataset.load_data(path)

    cols = dataset.categorical_columns + dataset.extra_categorical_columns
    df = dataset.get_data_set_for_analysis()

    df_encoded = pd.get_dummies(data=df, columns=cols)

    # add the prediction variable
    df_encoded['cat'] = df['cat']


    total_class_zero = df_encoded['cat'].where(lambda cat: cat == 0).count()
    total_class_one = df_encoded['cat'].where(lambda cat: cat == 1).count()
    min = np.min([total_class_zero, total_class_one])

    N = min * 2

    # Obtenemos los datos y unimos las clases
    class_zero = df_encoded.loc[df['cat'] == 0].head(int(N/2))
    class_one = df_encoded.loc[df['cat'] == 1].head(int(N/2))
    df_analysis = pd.concat([class_zero, class_one], ignore_index=True)

    # Revolvemos los datos aleatóriamente
    df_analysis = df_analysis.sample(frac=1).reset_index(drop=True)

    return df_analysis
