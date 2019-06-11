import random
import pandas as pd
from math import sqrt


def split_by_row(df, number_of_groups):
    """
    Split the dataset by rows into x number of groups
    :param df: the data frame with the data set
    :param number_of_groups: the number of groups to be split
    :return: the groups with the data set split
    """
    groups = []
    group_len = int(len(df) / number_of_groups)

    for i in range(0, number_of_groups):
        current_cutoff = i * group_len
        next_cutoff = current_cutoff + group_len

        if i + 1 == number_of_groups:
            groups.append(df[current_cutoff:].reset_index(drop=True))
        else:
            groups.append(df[current_cutoff:next_cutoff].reset_index(drop=True))

    return groups


def train_multiple(data, number_of_groups, max_iter=10):
    """
    train multiple kmeans
    :param data: the data
    :param number_of_groups: the number of groups
    :param max_iter: the max number of iterations
    :return: the list of kmeans
    """
    groups = split_by_row(data, number_of_groups)
    kmeans_list = []
    for group in groups:
        kmeans_classfier = Kmeans(group, 'cat')
        kmeans_classfier.train(max_iter=max_iter)
        kmeans_list.append(kmeans_classfier)

    return kmeans_list


def correctness_with_multiple_kmeans(test_df, kmeans_list):
    """
    find the correctness of trianing multiple datasets
    :param test_df: the test dataset
    :param kmeans_list: the list of kmeans
    :return: the correctness as scalar
    """
    data = test_df.reset_index(drop=True)

    corrects = 0
    for i in range(0, len(data)):
        series = data.loc[i]
        real_value = series['cat']
        series = series.drop(labels=['cat'])

        predictions_zero = 0
        predictions_one = 0
        for kmeans in kmeans_list:
            distance_with_zero = kmeans.distance(series, kmeans.centroid_zero)
            distance_with_one = kmeans.distance(series, kmeans.centroid_one)

            if distance_with_zero <= distance_with_one:
                predictions_zero += 1
            else:
                predictions_one += 1

        if predictions_zero >= predictions_one:
            prediction = 0
        else:
            prediction = 1

        if prediction == real_value:
            corrects += 1

    return float(corrects) / float(len(data))


class Kmeans:
    """A class to perform clustering using kmeans"""
    def __init__(self, df, label):
        """
        Initialize the kmeans class and initialize the centroids, in this case there are only two centrois
        one for 1 and one for 0.
        :param df: The pandas data frame with dataset
        :param label: the label column (variable to predict)
        """
        self.df = df
        self.label = label
        self.centroid_zero, self.centroid_one = self.get_initial_centroids()
        self.cluster_zero = []
        self.cluster_one = []

    def get_initial_centroids(self):
        """
        Initialize the two centroids: centroid for one and centroid for zero. Using a randomly selected sample
        :return: The two centroids as pandas series
        """
        zeros = self.df[self.df[self.label] == 0].reset_index(drop=True)
        zeros = zeros.drop(self.label, axis=1)
        centroid_zero = zeros.loc[random.sample(range(0, len(zeros)), 1)[0]]

        ones = self.df[self.df[self.label] == 1].reset_index(drop=True)
        ones = ones.drop(self.label, axis=1)
        centroid_one = ones.loc[random.sample(range(0, len(ones)), 1)[0]]

        return centroid_zero, centroid_one

    def distance(self, series, centroid):
        """
        Calculate the euclidean distance (norm l2) between a data point and a centroid.
        :param series: The series represents the data point
        :param centroid: The centroid which to calculate the distance
        :return: the distance as a float scalar
        """
        if len(series) != len(centroid):
            # For some reason (possible a bug) sometimes the series is emtpy and it was throwing this exception,
            # that's why I commented it.
            #raise Exception('Incorrect size')
            return 0

        dist = 0
        for i in range(len(series)):
            dist += (float(centroid[i]) - float(series[i])) ** 2

        return sqrt(dist)

    def iterate(self, data):
        """
        Each iteration it will loop through all data and find the distance against the two centroids and
        find which one is closer to, based on that add the cluster into one bucket or the other (cluster zero or
        cluster one)
        :param data: The data frame representing the dataset
        """
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

    def train(self, max_iter=10):
        """
        Train the cluster using kmeans. The result of this training will basically be the location in all dimensions
        of the two centroids.
        :param max_iter: Maximum number of iterations before stop running the algorithm
        """
        # Drop the label
        data = self.df.drop(self.label, axis=1).reset_index(drop=True)

        iteration = 0
        while iteration < max_iter:
            self.iterate(data)
            iteration += 1

    def test_correctness(self, data):
        """
        Test the correctness of inference using kmeans clustering for classification.
        :param data: The data frame representing the dataset
        :return: A number between 0 and 1 representing the correctness of the prediction where 1 means the prediction
        is totally correct and 0 is that prediction is totally incorrect.
        """
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
# from sklearn.model_selection import train_test_split
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


