import random
import pandas as pd

from math import sqrt
from sklearn.model_selection import train_test_split
import dataset


def get_initial_centroids(df):
    zeros = df[train['cat'] == 0].reset_index(drop=True)
    zeros = zeros.drop('cat', axis=1)
    centroid_zero = zeros.loc[random.sample(range(0, len(zeros)), 1)[0]]

    ones = df[train['cat'] == 1].reset_index(drop=True)
    ones = ones.drop('cat', axis=1)
    centroid_one = ones.loc[random.sample(range(0, len(ones)), 1)[0]]

    return centroid_zero, centroid_one


def distance(series, centroid):
    if len(series) != len(centroid):
        raise Exception('Incorrect size')

    dist = 0
    for i in range(len(series)):
        dist += (float(centroid[i]) - float(series[i])) ** 2

    return sqrt(dist)


def iterate(data, centroid_zero, centroid_one):
    cluster_zero = []
    cluster_one = []

    for i in range(0, len(data)):
        series = data.loc[i]
        distance_with_zero = distance(series, centroid_zero)
        distance_with_one = distance(series, centroid_one)

        if distance_with_zero <= distance_with_one:
            cluster_zero.append(series)
        else:
            cluster_one.append(series)

    centroid_zero = pd.DataFrame(cluster_zero).mean(axis=0)
    centroid_one = pd.DataFrame(cluster_one).mean(axis=0)
    return centroid_zero, centroid_one


def run_kmeans(data, max_iter=10):
    centroid_zero, centroid_one = get_initial_centroids(data)
    data = data.drop('cat', axis=1).reset_index(drop=True)

    iteration = 0
    while iteration < max_iter:
        centroid_zero, centroid_one = iterate(data, centroid_zero, centroid_one)
        iteration += 1

    return centroid_zero, centroid_one


def correctness(data, centroid_zero, centroid_one):
    corrects = 0
    data = data.reset_index(drop=True)
    for i in range(0, len(data)):
        series = data.loc[i]
        real_value = series['cat']
        series = series.drop(labels=['cat'])

        distance_with_zero = distance(series, centroid_zero)
        distance_with_one = distance(series, centroid_one)

        if distance_with_zero <= distance_with_one:
            prediction = 0
        else:
            prediction = 1

        if prediction == real_value:
            corrects += 1

    return float(corrects) / float(len(data))


df = dataset.get_data_for_model('data/construction-data-processed.csv', balanced=False)
df = df.head(1000)
train, test = train_test_split(df, test_size=0.1)

zero, one = run_kmeans(train, 10)

print(correctness(test, zero, one))


