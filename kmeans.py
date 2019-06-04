import random
import pandas as pd

from math import sqrt
from sklearn.model_selection import train_test_split


class Kmeans:
    def __init__(self, df, label):
        self.df = df
        self.label = label
        self.centroid_zero, self.centroid_one = self.get_initial_centroids()
        self.cluster_zero = []
        self.cluster_one = []

    def get_initial_centroids(self):
        zeros = self.df[self.df[self.label] == 0].reset_index(drop=True)
        zeros = zeros.drop(self.label, axis=1)
        centroid_zero = zeros.loc[random.sample(range(0, len(zeros)), 1)[0]]

        ones = self.df[self.df[self.label] == 1].reset_index(drop=True)
        ones = ones.drop(self.label, axis=1)
        centroid_one = ones.loc[random.sample(range(0, len(ones)), 1)[0]]

        return centroid_zero, centroid_one

    def distance(self, series, centroid):
        if len(series) != len(centroid):
            return 0
            #raise Exception('Incorrect size')

        dist = 0
        for i in range(len(series)):
            dist += (float(centroid[i]) - float(series[i])) ** 2

        return sqrt(dist)

    def iterate(self, data):
        self.cluster_zero = []
        self.cluster_one = []

        for i in range(0, len(data)):
            series = data.loc[i]
            distance_with_zero = self.distance(series, self.centroid_zero)
            distance_with_one = self.distance(series, self.centroid_one)

            if distance_with_zero <= distance_with_one:
                self.cluster_zero.append(series)
            else:
                self.cluster_one.append(series)

        self.centroid_zero = pd.DataFrame(self.cluster_zero).mean(axis=0)
        self.centroid_one = pd.DataFrame(self.cluster_one).mean(axis=0)

    def train(self, data, max_iter=10):
        # Drop the label
        data = data.drop(self.label, axis=1).reset_index(drop=True)

        iteration = 0
        while iteration < max_iter:
            self.iterate(data)
            iteration += 1

    def test_correctness(self, data):
        data = data.reset_index(drop=True)

        corrects = 0
        for i in range(0, len(data)):
            series = data.loc[i]
            real_value = series['cat']
            series = series.drop(labels=['cat'])

            distance_with_zero = self.distance(series, self.centroid_zero)
            distance_with_one = self.distance(series, self.centroid_one)

            if distance_with_zero <= distance_with_one:
                prediction = 0
            else:
                prediction = 1

            if prediction == real_value:
                corrects += 1

        return float(corrects) / float(len(data))


# import dataset
# import utilities
#
# df = dataset.get_data_for_model('data/construction-data-processed.csv', balanced=False)
# #df = df[['cat', 'numviv_cat_1', 'arecon_cat_50']].head(1000)
# df = df.head(1000)
# train, test = train_test_split(df, test_size=0.1)

#
# def train_multiple(data, number_of_groups, max_iter=10):
#     groups = utilities.split_by_row(data, number_of_groups)
#     kmeans_list = []
#     for group in groups:
#         kmeans = Kmeans(group, 'cat')
#         kmeans.train(train, max_iter=max_iter)
#         kmeans_list.append(kmeans)
#
#     return kmeans_list
#
#
# def correctness_with_multiple_kmeans(test_df, kmeans_list):
#     data = test_df.reset_index(drop=True)
#
#     corrects = 0
#     for i in range(0, len(data)):
#         series = data.loc[i]
#         real_value = series['cat']
#         series = series.drop(labels=['cat'])
#
#         predictions_zero = 0
#         predictions_one = 0
#         for kmeans in kmeans_list:
#             distance_with_zero = kmeans.distance(series, kmeans.centroid_zero)
#             distance_with_one = kmeans.distance(series, kmeans.centroid_one)
#
#             if distance_with_zero <= distance_with_one:
#                 predictions_zero += 1
#             else:
#                 predictions_one += 1
#
#         if predictions_zero >= predictions_one:
#             prediction = 0
#         else:
#             prediction = 1
#
#         if prediction == real_value:
#             corrects += 1
#
#     return float(corrects) / float(len(data))
#
#
# print('Starting training:')
# kmeans_list = train_multiple(train, 3, max_iter=5)
#
# print('Starting testing:')
# correctness = correctness_with_multiple_kmeans(test, kmeans_list)
# print(correctness)


# print('Starting to process:')
# kmeans = Kmeans(train, 'cat')
# kmeans.train(train, max_iter=5)
# print(kmeans.test_correctness(test))


