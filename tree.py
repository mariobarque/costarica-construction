import pandas as pd
from sklearn.model_selection import train_test_split
from math import log

class Node:
    def __init__(self, feature, true_branch, false_branch, impurity, info_gain):
        self.feature = feature
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.impurity = impurity
        self.info_gain = info_gain

    def print(self):
        self.print_tree(self)

    def print_tree(self, node, spacing=""):
        if isinstance(node, Leaf):
            print('%sPrediction: %s' % (spacing, node.print()))
            return

        print('%s-->%s; impurity: %f; info_gain: %f --> True: ' % (spacing, str(node.feature), node.impurity, node.info_gain))
        self.print_tree(node.true_branch, spacing + "  ")

        print('%s-->%s; impurity: %f; info_gain: %f --> False: ' % (spacing, str(node.feature), node.impurity, node.info_gain))
        self.print_tree(node.false_branch, spacing + "  ")

    def predict_dataset(self, df):
        prediction = []
        for _, row in df.iterrows():
            prediction.append(self.predict(row, self))

        return prediction

    def predict(self, row, node):
        if isinstance(node, Leaf):
            return node.value

        if row[node.feature] == 0:
            return self.predict(row, node.false_branch)
        else:
            return self.predict(row, node.true_branch)

class Leaf:
    def __init__(self, label_column):
        class_counts = label_column.value_counts()
        if len(class_counts) == 0:
            most_important_class = 0
        else:
            most_important_class = class_counts.index[0]

        self.plurality = dict(class_counts)
        self.value = most_important_class

    def print(self):
        return "%d : %s" % (self.value, str(self.plurality))


class TreeBuilder:
    def __init__(self, label, impurity_function = 'entropy'):
        self.label = label

        if impurity_function == 'entropy':
            self.impurity_fuc = self.entropy
        else:
            if impurity_function == 'gini':
                self.impurity_fuc = self.gini
            else:
                raise Exception('Invalid impurity function')


    def gini(self, df):
        counts = list(df[self.label].value_counts())
        impurity = 1
        for lbl in counts:
            prob_of_lbl = lbl / float(len(df))
            impurity -= prob_of_lbl ** 2
        return impurity


    def entropy(self, df):
        counts = list(df[self.label].value_counts())
        impurity = 0
        for cat_class in counts:
            prob_of_class = cat_class / float(len(df))
            impurity -= prob_of_class * log(prob_of_class, 2)
        return impurity


    def info_gain(self, left, right, impurity):
        p = 0
        if len(left) + len(right) > 0:
            p = float(len(left)) / (len(left) + len(right))
        new_uncertainty = p * self.impurity_fuc(left) - (1 - p) * self.impurity_fuc(right)
        return impurity - new_uncertainty


    @staticmethod
    def partition(df, column):
        if column is None:
            return pd.DataFrame(data=None), df

        true_rows, false_rows = df.loc[df[column] == 1], df.loc[df[column] == 0]
        return true_rows, false_rows


    def find_best_feature(self, df, features):
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


    def build_tree_cart(self, df):
        features = list(df.columns.values)[1:]
        impurity, info_gain, feature = self.find_best_feature(df, features)

        if info_gain == 0:
            return Leaf(df[self.label])

        true_df, false_df = TreeBuilder.partition(df, feature)

        true_branch = self.build_tree_cart(true_df)
        false_branch = self.build_tree_cart(false_df)

        return Node(feature, true_branch, false_branch, impurity, info_gain)


    def build_tree_id3(self, df):
        features = list(df.columns.values)[1:]
        return self.build_tree_id3_rec(features, df, df)


    def get_classes(self, df):
        class_counts = df[self.label].value_counts()
        return class_counts


    def build_tree_id3_rec(self, features, df, parent_df):
        if len(df) == 0:
            return Leaf(parent_df[self.label])

        if len(features) == 0:
            return Leaf(df[self.label])

        classes = self.get_classes(df)
        if len(classes) == 1:
            return Leaf(df[self.label])

        impurity, info_gain, best_feature = self.find_best_feature(df, features)

        true_df, false_df = TreeBuilder.partition(df, best_feature)
        if best_feature is not None:
            features.remove(best_feature)

        # Because I know there are only two classes then I don't need to worry about
        # Looping through all possible classes.
        true_branch = self.build_tree_id3_rec(features, true_df, df)
        false_branch = self.build_tree_id3_rec(features, false_df, df)

        return Node(best_feature, true_branch, false_branch, impurity, info_gain)

class RandomForest:
    def __init__(self, df, label, col_groups, row_groups, train_test_ratio=0.01, impurity_function='entropy', algorithm='id3'):
        self.label = label
        self.train_test_ratio = train_test_ratio
        self.train, self.evaluation_test = train_test_split(df, test_size=self.train_test_ratio)
        self.groups = self.create_groups(col_groups, row_groups)
        self.trees = []
        self.algorithm = algorithm
        self.tree_builder = TreeBuilder(self.label, impurity_function)

    @staticmethod
    def split_by_row(df, number_of_groups):
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
        col_groups = self.split_by_col(col_groups_count)
        groups = []
        for col_group in col_groups:
            row_groups = RandomForest.split_by_row(col_group, row_groups_count)
            [groups.append(g) for g in row_groups]

        return groups


    def train_forest(self, top=None):
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
            self.trees.append(tree)


    def predict(self, row, node):
        if isinstance(node, Leaf):
            return node.value

        if row[node.feature] == 0:
            return self.predict(row, node.false_branch)
        else:
            return self.predict(row, node.true_branch)


    '''
    Find the consensus between all trees to find the prediction 
    '''
    def predict_row(self, row):
        all_tree_predictions = []
        for tree in self.trees:
            tree_prediction = tree.predict(row, tree)
            all_tree_predictions.append(tree_prediction)

        if all_tree_predictions.count(1) > all_tree_predictions.count(0):
            return 1
        else:
            return 0

    def predict_test(self):
        prediction = []
        for row in self.evaluation_test:
            prediction.append(self.predict_row(row))

        return prediction

    def evaluate(self, df):
        correct_predictions = self.get_correct_predictions(df)

        # Return the percentage of correct predictions
        return float(correct_predictions)/float(len(df))

    def get_correct_predictions(self, df):
        correct_predictions = 0
        for _, row in df.iterrows():
            prediction = self.predict_row(row)
            real = row[self.label]
            if prediction == real:
                correct_predictions += 1

        return correct_predictions

    def evaluate_forest(self, max_size = 100, increase = 1):
        current_size = 0
        result = pd.DataFrame()
        while current_size <= max_size:
            current_size += increase

            self.train_forest(top=current_size)

            correctness = self.evaluate(self.evaluation_test)

            val = {'training_size': current_size, 'correctness': correctness}
            print(val)
            result.append(val, ignore_index=True)

        return result
