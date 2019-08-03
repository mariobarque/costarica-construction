from sklearn.model_selection import train_test_split

import dataset
from tree import tree_utilities
from tree.leaf import Leaf
from tree.node import Node
from math import log
import pandas as pd


# class TreeBuilder:
#     def __init__(self, label, impurity_function = 'entropy'):
#         self.label = label
#
#         if impurity_function == 'entropy':
#             self.impurity_fuc = self.entropy
#         else:
#             if impurity_function == 'gini':
#                 self.impurity_fuc = self.gini
#             else:
#                 raise Exception('Invalid impurity function')
#
#
#     def gini(self, df):
#         counts = list(df[self.label].value_counts())
#         impurity = 1
#         for lbl in counts:
#             prob_of_lbl = lbl / float(len(df))
#             impurity -= prob_of_lbl ** 2
#         return impurity
#
#
#     def entropy(self, df):
#         counts = list(df[self.label].value_counts())
#         impurity = 0
#         for cat_class in counts:
#             prob_of_class = cat_class / float(len(df))
#             impurity += prob_of_class * -1 * log(prob_of_class, 2)
#         return impurity
#
#
#     def info_gain(self, left, right, uncertainty):
#         p = 0
#         if len(left) + len(right) > 0:
#             p = float(len(left)) / (len(left) + len(right))
#         new_uncertainty = p * self.impurity_fuc(left) - (1 - p) * self.impurity_fuc(right)
#         return uncertainty - new_uncertainty
#
#
#     @staticmethod
#     def partition(df, column):
#         if column is None:
#             return pd.DataFrame(data=None), df
#
#         true_rows, false_rows = df.loc[df[column] == 1], df.loc[df[column] == 0]
#         return true_rows, false_rows
#
#
#     def find_best_feature(self, df, features):
#         best_gain = 0
#         best_feature = features[0]
#         current_uncertainty = self.impurity_fuc(df)
#
#         for feature in features:
#             true_rows, false_rows = TreeBuilder.partition(df, feature)
#
#             if len(true_rows) == 0 or len(false_rows) == 0:
#                 continue
#
#             current_gain = self.info_gain(true_rows, false_rows, current_uncertainty)
#
#
#             if current_gain >= best_gain:
#                 best_gain, best_feature = current_gain, feature
#
#         if best_feature is None:
#             print('This is an issue')
#
#         return best_gain, best_feature
#
#
#     def build_tree_cart(self, df):
#         features = list(df.columns.values)[1:]
#         current_gain, feature = self.find_best_feature(df, features)
#
#         if current_gain == 0:
#             return Leaf(df[self.label])
#
#         true_df, false_df = TreeBuilder.partition(df, feature)
#
#         true_branch = self.build_tree_cart(true_df)
#         false_branch = self.build_tree_cart(false_df)
#
#         return Node(feature, true_branch, false_branch)
#
#
#     def build_tree_id3(self, df):
#         features = list(df.columns.values)[1:]
#         return self.build_tree_id3_rec(features, df, df)
#
#
#     def get_classes(self, df):
#         class_counts = df[self.label].value_counts()
#         return class_counts
#
#
#     def build_tree_id3_rec(self, features, df, parent_df):
#         if len(df) == 0:
#             return Leaf(parent_df[self.label])
#
#         if len(features) == 0:
#             return Leaf(df[self.label])
#
#         classes = self.get_classes(df)
#         if len(classes) == 1:
#             return Leaf(df[self.label])
#
#         _, best_feature = self.find_best_feature(df, features)
#
#         true_df, false_df = TreeBuilder.partition(df, best_feature)
#         if best_feature is not None:
#             features.remove(best_feature)
#
#         # Because I know there are only two classes then I don't need to worry about
#         # Looping through all possible classes.
#         true_branch = self.build_tree_id3_rec(features, true_df, df)
#         false_branch = self.build_tree_id3_rec(features, false_df, df)
#
#         return Node(best_feature, true_branch, false_branch)




###########################################################################################
###########################################################################################
#########################From here on its testing only#####################################
##################################Ignore this##############################################
###########################################################################################


#data = dataset.get_data_for_model('../data/construction-data-processed.csv', balanced=False)

#cols_to_remove = ['anoper_2002','anoper_2003','anoper_2004','anoper_2005','anoper_2006','anoper_2007','anoper_2009','anoper_2010','anoper_2011','anoper_2012','anoper_2013','anoper_2014','anoper_2015','anoper_2016','anoper_2017','anoper_2018', 'financ_99', 'matpis_1','matpis_2','matpis_4','matpis_5','matpis_6','matpis_7','matpis_8','matpis_9','matpis_10','matpis_11','matpis_12','matpis_13','matpis_14','matpis_15','matpis_16','matpis_17','matpis_18','matpis_19','matpis_20','matpis_21','matpis_22','matpis_23','matpis_24','matpis_25','matpis_26','matpis_27','matpis_28','matpis_29','matpis_30','matpis_31','matpis_32','matpis_33','matpis_34','matpis_88','matpis_98','matpis_99']
#data = data.head(1000)
#data = data.drop(cols_to_remove, axis=1)
#print(data)
#train, test = train_test_split(data, test_size=0.1)

#data = pd.read_csv('data.csv')

#tree_builder = TreeBuilder('cat', 'entropy')

#tree = tree_builder.build_tree_id3(train)
#tree_utilities.print_tree(tree)
#print(tree_utilities.predict(test, tree))

#true_rows, false_rows = TreeBuilder.partition(train, 'matpis_8')
#true_branch, false_branch = Leaf(true_rows['cat']), Leaf(false_rows['cat'])
#true_entropy, false_entropy = tree_builder.entropy(true_rows), tree_builder.entropy(false_rows)
#print('TRUE:' + true_branch.print() + ', entorpy: ' + str(true_entropy)
#      + ' ||| FALSE:' + false_branch.print() + ', entorpy: ' + str(false_entropy))



#features = list(data.columns.values)[1:]
#tree_builder = TreeBuilder('cat', 'gini')
#tree_builder.find_best_feature(data, features)

#current_gain, feature = self.find_best_feature(df, features)