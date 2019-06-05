import pandas as pd
from sklearn.model_selection import train_test_split
from math import log


class Node:
    """ Represents a node in a decision tree """
    def __init__(self, feature, true_branch, false_branch, impurity, info_gain):
        """
        Initialize a node for a decision tree

        :param feature: The feature by which the node is split
        :param true_branch: The true branch which means the branch that satisfies the condition
        :param false_branch: The false branch which means the branch that does not satisfies the condition
        :param impurity: The impurity (gini or entropy) used for printing the tree
        :param info_gain: The information gain used for printing the tree
        """

        self.feature = feature
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.impurity = impurity
        self.info_gain = info_gain

    def print(self):
        """
        Prints a tree
        """
        self.print_tree(self)

    def print_tree(self, node, spacing=""):
        """
        Prints a tree
        :param node: The root node to start printing the tree
        :param spacing: the spacing by default blank space
        """
        if isinstance(node, Leaf):
            print('%sPrediction: %s' % (spacing, node.print()))
            return

        print('%s-->%s; impurity: %f; info_gain: %f --> True: ' % (spacing, str(node.feature), node.impurity, node.info_gain))
        self.print_tree(node.true_branch, spacing + "  ")

        print('%s-->%s; impurity: %f; info_gain: %f --> False: ' % (spacing, str(node.feature), node.impurity, node.info_gain))
        self.print_tree(node.false_branch, spacing + "  ")

    def predict_dataset(self, df):
        """
        Predict a dataset based on the tree where self node is the root
        :param df: the dataset to predict
        :return: returns the prediction as an array
        """
        prediction = []
        for _, row in df.iterrows():
            prediction.append(self.predict(row, self))

        return prediction

    def predict(self, row, node):
        """
        Predict a single sample
        :param row: the row that contains all attributes in the sample
        :param node: the node that represents the tree (it's used for recursion)
        :return: the prediction as an scalar
        """
        if isinstance(node, Leaf):
            return node.value

        if row[node.feature] == 0:
            return self.predict(row, node.false_branch)
        else:
            return self.predict(row, node.true_branch)


class Leaf:
    """ Represents a leaf in a decision tree """
    def __init__(self, label_column):
        """
        Initialize the leaf in the decision tree
        :param label_column: the label column (variable to predict)
        """
        class_counts = label_column.value_counts()
        if len(class_counts) == 0:
            most_important_class = 0
        else:
            most_important_class = class_counts.index[0]

        self.plurality = dict(class_counts)
        self.value = most_important_class

    def print(self):
        """
        Print a leaf
        :return: the string with the leaf representation
        """
        return "%d : %s" % (self.value, str(self.plurality))


class TreeBuilder:
    """ This class contains utilities to build a decision tree """

    def __init__(self, label, impurity_function='entropy'):
        """
        Initialize the tree builder
        :param label: the label column (variable to predict)
        :param impurity_function: either gini or entropy.
        """
        self.label = label

        if impurity_function == 'entropy':
            self.impurity_fuc = self.entropy
        else:
            if impurity_function == 'gini':
                self.impurity_fuc = self.gini
            else:
                raise Exception('Invalid impurity function')

    def gini(self, df):
        """
        Calculate the impurity function based on gini function
        :param df: The pandas data frame with dataset to calculate the impurity
        :return: The impurity as scalar.
        """
        counts = list(df[self.label].value_counts())
        impurity = 1
        for lbl in counts:
            prob_of_lbl = lbl / float(len(df))
            impurity -= prob_of_lbl ** 2
        return impurity

    # @staticmethod
    # def gini(counts, length):
    #     impurity = 1
    #     for lbl in counts:
    #         prob_of_lbl = lbl / float(length)
    #         impurity -= prob_of_lbl ** 2
    #     return impurity

    def entropy(self, df):
        """
        Calculates the impurity using entropy function
        :param df: The pandas data frame with dataset to calculate the impurity
        :return: The impurity as scalar.
        """
        counts = list(df[self.label].value_counts())
        impurity = 0
        for cat_class in counts:
            prob_of_class = cat_class / float(len(df))
            impurity -= prob_of_class * log(prob_of_class, 2)
        return impurity

    # @staticmethod
    # def entropy(counts, length):
    #     impurity = 0
    #     for cat_class in counts:
    #         prob_of_class = cat_class / float(length)
    #         impurity -= prob_of_class * log(prob_of_class, 2)
    #     return impurity

    def info_gain(self, left, right, impurity):
        """
        Calculates the information gain based on the two children datasets and the previous impurity
        :param left: The left child dataset
        :param right: The right child dataset
        :param impurity: The previous impurity
        :return: the information gain as scalar
        """
        p = 0
        if len(left) + len(right) > 0:
            p = float(len(left)) / (len(left) + len(right))
        new_uncertainty = p * self.impurity_fuc(left) - (1 - p) * self.impurity_fuc(right)
        return impurity - new_uncertainty

    # @staticmethod
    # def info_gain(left_length, right_length, impurity_left, impurity_right, impurity):
    #     p = 0
    #     if left_length + right_length > 0:
    #         p = float(left_length) / (left_length + right_length)
    #     new_uncertainty = p * impurity_left - (1 - p) * impurity_right
    #     return impurity - new_uncertainty

    @staticmethod
    def partition(df, feature):
        """
        Partition a pandas data frame into two subsets based on a column (feature)
        :param df: The pandas data frame to partition
        :param feature: The feature by which to partition the dataset
        :return: two data frame one for true values and the other for false values.
        """
        if feature is None:
            return pd.DataFrame(data=None), df

        true_rows, false_rows = df.loc[df[feature] == 1], df.loc[df[feature] == 0]
        return true_rows, false_rows

    def find_best_feature(self, df, features):
        """
        Find the best feature based to partition the dataset to find the next node in the decision tree.
        The best feature is identified as the one that gets more information gain.
        :param df: The pandas data frame with the dataset
        :param features: All the features (columns) from where to find the best one
        :return: The impurity of that the best feature, the information gain and the best feature
        """
        best_gain = 0
        best_feature = features[0]
        impurity = self.impurity_fuc(df)

        for feature in features:
            true_rows, false_rows = TreeBuilder.partition(df, feature)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            current_gain = self.info_gain(true_rows, false_rows, impurity)

            if current_gain >= best_gain:
                best_gain, best_feature = current_gain, feature

        return impurity, best_gain, best_feature

    def get_classes(self, df):
        """
        Get different classes, in this case it would be ones and zeros.
        :param df: The pandas data frame with the dataset
        :return: the classes, {ones: quantity of ones, zeros: quantity of zeros}
        """
        class_counts = df[self.label].value_counts()
        return class_counts

    def build_tree_cart(self, df):
        """
        Build a decision tree using cart algorithm
        :param df: The pandas data frame with the dataset
        :return: The decision tree
        """
        features = list(df.columns.values)[1:]
        impurity, info_gain, feature = self.find_best_feature(df, features)

        if info_gain == 0:
            return Leaf(df[self.label])

        true_df, false_df = TreeBuilder.partition(df, feature)

        true_branch = self.build_tree_cart(true_df)
        false_branch = self.build_tree_cart(false_df)

        return Node(feature, true_branch, false_branch, impurity, info_gain)

    def build_tree_id3(self, df):
        """
        Builds a decision tree using id3 algorithm.
        :param df: The pandas data frame with the dataset
        :return: The decision tree
        """
        features = list(df.columns.values)[1:]
        return self.build_tree_id3_rec(features, df, df)

    def build_tree_id3_rec(self, features, df, parent_df, prune_when_info_gain_less_than=0.5):
        """
        The recursion method used to build the tree using id3 algorithm
        :param features: The features (columns) in the dataset
        :param df: The pandas data frame with the dataset
        :param parent_df: The parent dataset
        :param prune_when_info_gain_less_than: A measure to prune nodes/leafs when information gain is less a threshold
        :return: The decision tree
        """
        if len(df) == 0:
            return Leaf(parent_df[self.label])

        if len(features) == 0:
            return Leaf(df[self.label])

        classes = self.get_classes(df)
        if len(classes) == 1:
            return Leaf(df[self.label])

        impurity, info_gain, best_feature = self.find_best_feature(df, features)

        # Pruning
        if info_gain < prune_when_info_gain_less_than:
            features.remove(best_feature)
            return Leaf(df[self.label])

        true_df, false_df = TreeBuilder.partition(df, best_feature)
        if best_feature is not None:
            features.remove(best_feature)

        # Because I know there are only two classes then I don't need to worry about
        # Looping through all possible classes.
        true_branch = self.build_tree_id3_rec(features, true_df, df)
        false_branch = self.build_tree_id3_rec(features, false_df, df)

        return Node(best_feature, true_branch, false_branch, impurity, info_gain)


class RandomForest:
    """ A class to create random forests """
    def __init__(self, df, label, col_groups, row_groups, train_test_ratio=0.01, impurity_function='entropy', algorithm='id3'):
        """
        Initializes the random forest
        :param df: The pandas data frame with the dataset
        :param label: the label (prediction variable)
        :param col_groups: The number of groups split by column
        :param row_groups: The number of groups split by row
        :param train_test_ratio: The test and train ration
        :param impurity_function: The impurity function: 'entropy', 'gini'. By default 'entropy'
        :param algorithm: The algorithm to use: 'cart' or 'id3'. By default 'id3'
        """
        self.label = label
        self.train_test_ratio = train_test_ratio
        self.train, self.evaluation_test = train_test_split(df, test_size=self.train_test_ratio)
        self.groups = self.create_groups(col_groups, row_groups)
        self.trees = []
        self.algorithm = algorithm
        self.tree_builder = TreeBuilder(self.label, impurity_function)

    @staticmethod
    def split_by_row(df, number_of_groups):
        """
        Split dataset by rows
        :param df:  The pandas data frame with the dataset
        :param number_of_groups: The number of groups
        :return: A list of datasets with the groups
        """
        groups = []
        group_len = int(len(df)/number_of_groups)

        for i in range(0, number_of_groups):
            current_cutoff = i * group_len
            next_cutoff = current_cutoff + group_len

            if i + 1 == number_of_groups:
                groups.append(df[current_cutoff:].reset_index(drop=True))
            else:
                groups.append(df[current_cutoff:next_cutoff].reset_index(drop=True))

        return groups

    def split_by_col(self, number_of_groups):
        """
        Split dataset by columns
        :param number_of_groups: The number of groups
        :return: A list of datasets with the groups
        """
        all_cols_without_label = list(self.train.columns.values)[1:]
        groups = []
        group_len = int(len(all_cols_without_label) / number_of_groups)

        for i in range(0, number_of_groups):
            current_cutoff = i * group_len
            next_cutoff = current_cutoff + group_len

            new_columns = [self.label]

            if i + 1 == number_of_groups:
                [new_columns.append(c) for c in all_cols_without_label[current_cutoff:]]
                groups.append(self.train[new_columns].copy())
            else:
                [new_columns.append(c) for c in all_cols_without_label[current_cutoff:next_cutoff]]
                groups.append(self.train[new_columns].copy())

        return groups

    def create_groups(self, col_groups_count, row_groups_count):
        """
        Creates groups based on the number of columns groups and the number of rows groups
        It splits the dataset initially by columns and out of every column group a number of row group is created.
        :param col_groups_count: The number of column groups
        :param row_groups_count: The number of row groups
        :return: Return the list of groups
        """
        col_groups = self.split_by_col(col_groups_count)
        groups = []
        for col_group in col_groups:
            row_groups = RandomForest.split_by_row(col_group, row_groups_count)
            [groups.append(g) for g in row_groups]

        return groups

    def train_forest(self, top=None):
        """
        Train the random forest
        For each group create a decision tree and adds the tree into a tree array.
        :param top: The top represents how many samples to train in the random forest,
                    It's use for creating a metric when increasing the number of samples and check how well it's
                    predicting
        :return: An array of decision trees (the random forest)
        """
        self.trees.clear()
        for group in self.groups:
            if self.algorithm == 'id3':
                if top is None:
                    tree = self.tree_builder.build_tree_id3(group)
                else:
                    tree = self.tree_builder.build_tree_id3(group.head(top))
            else:
                if self.algorithm == 'cart':
                    if top is None:
                        tree = self.tree_builder.build_tree_cart(group)
                    else:
                        tree = self.tree_builder.build_tree_cart(group.head(top))
                else:
                    raise Exception('Invalid algorithm')

            if not isinstance(tree, Leaf):
                self.trees.append(tree)

    def predict(self, row, node):
        """
        Predict a single row using a decision tree represented by node. It uses recursion
        :param row: The row representing sample with all its features
        :param node: The node representing the decision tree
        :return: The prediction as scalar.
        """
        if isinstance(node, Leaf):
            return node.value

        if row[node.feature] == 0:
            return self.predict(row, node.false_branch)
        else:
            return self.predict(row, node.true_branch)

    def predict_row(self, row):
        """
        Predict a single sample using the decision tree.
        Find the consensus between all trees to find the prediction
        :param row: The row representing sample with all its features
        :return: The prediction as scalar
        """
        all_tree_predictions = []
        for tree in self.trees:
            tree_prediction = tree.predict(row, tree)
            all_tree_predictions.append(tree_prediction)

        if all_tree_predictions.count(1) > all_tree_predictions.count(0):
            return 1
        else:
            return 0

    def predict_test(self):
        """
        Test the random forest by predicting with an evaluation test samples.
        :return: The prediction as an array of values
        """
        prediction = []
        for row in self.evaluation_test:
            prediction.append(self.predict_row(row))

        return prediction

    def evaluate(self, df):
        """
        Evaluates the correctness of a prediction using random forest
        :param df: The pandas data frame with the dataset
        :return: The correctness with a number between 0 and 1.
        """
        correct_predictions = self.get_correct_predictions(df)

        # Return the percentage of correct predictions
        return float(correct_predictions)/float(len(df))

    def get_correct_predictions(self, df):
        """
        Get the correct predictions
        :param df: The pandas data frame with the dataset
        :return: the number of correct predictions as scalar
        """
        correct_predictions = 0
        for _, row in df.iterrows():
            prediction = self.predict_row(row)
            real = row[self.label]
            if prediction == real:
                correct_predictions += 1

        return correct_predictions

    def evaluate_forest(self, max_size=100, increase=1):
        """
        Evaluate the forest increase the size of samples (not trees) and evaluating in each iteration
        :param max_size: The max size of samples for the evaluation
        :param increase: How many samples are added in every iteration
        :return: A pandas data frame with the prediction that later on will be used to calculate correctness.
        """
        current_size = 0
        result = []
        while current_size <= max_size:
            current_size += increase

            self.train_forest(top=current_size)

            correctness = self.evaluate(self.evaluation_test)

            val = {'training_size': current_size, 'correctness': correctness}
            print(val)
            result.append(val)

        return pd.DataFrame(result)


###########################################################################################
###########################################################################################
#########################From here on its testing only#####################################
##################################Ignore this##############################################
###########################################################################################



#x = pd.read_csv('data.csv')
#
# import dataset
# data = dataset.get_data_for_model('data/construction-data-processed.csv', balanced=False)
# data = data.head(1000)
# train, test = train_test_split(data, test_size=0.1)
# #
# #
# tree_builder = TreeBuilder('cat', impurity_function='entropy')
# tree = tree_builder.build_tree_id3(train)
# prediction = tree.predict_dataset(test)
# tree.print()
# #
# corrects = 0
# for i in range(len(test)):
#     if test['cat'].values[i] == prediction[i]:
#         corrects += 1
#
# print(float(corrects) / float(len(test)))
#
#
#
#
#
#
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(criterion='entropy')
#
# x = train.drop('cat', axis=1)
# y = train['cat'].values
# model.fit(x, y)
#
# test_x = test.drop('cat', axis=1)
# test_y = test['cat'].values
#
# test_predict = model.predict(test_x)
#
# corrects = 0
# for i in range(len(test_y)):
#   if test_y[i] == test_predict[i]:
#     corrects += 1
#
# print(float(corrects) / float(len(test)))



# rf = RandomForest(train, 'cat', col_groups=12,
#                                               row_groups=2,
#                                               train_test_ratio=0.2,
#                                               impurity_function='entropy', # gini or entropy
#                                               algorithm='id3') # id3 or cart
# print('Starting to evaluate forest')
# evaluation = rf.evaluate_forest(increase=2, max_size=20)
# print(evaluation)