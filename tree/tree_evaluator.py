# from sklearn.model_selection import train_test_split
#
# import dataset
# from tree.leaf import Leaf
# import pandas as pd
#
# from tree.tree_builder import TreeBuilder
#
#
# class TreeEvaluator:
#     def __init__(self, tree_builder):
#         self.tree_builder = tree_builder
#
#     def build_tree(self, df, algorithm):
#         if algorithm == 'id3':
#             tree = self.tree_builder.build_tree_id3(df)
#         else:
#             if algorithm == 'cart':
#                 tree = self.tree_builder.build_tree_cart(df)
#             else:
#                 raise Exception('Invalid algorithm')
#
#         return tree
#
#     def predict(self, row, node):
#         if isinstance(node, Leaf):
#             return node.value
#
#         if row[node.feature] == 0:
#             return self.predict(row, node.false_branch)
#         else:
#             return self.predict(row, node.true_branch)
#
#     def get_correct_predictions(self, df, node):
#         correct_predictions = 0
#         for _, row in df.iterrows():
#             prediction = self.predict(row, node)
#             real = row[self.tree_builder.label]
#             if prediction == real:
#                 correct_predictions += 1
#
#         return correct_predictions
#
#     def evaluate_one_size(self, tree, df):
#         correct_predictions = self.get_correct_predictions(df, tree)
#
#         # Return the percentage of correct predictions
#         return float(correct_predictions)/float(len(df))
#
#     def evaluate(self, train, test, max_size = 100, increase = 1, algorithm='id3'):
#         if len(train) < max_size:
#             raise Exception('Not able to test up to %d, because data set size is %d'% (max_size, len(train)))
#
#         current_size = 0
#         result = pd.DataFrame()
#         while current_size <= max_size:
#             current_size += increase
#             print('Current Size: ' + str(current_size))
#
#             training_set = train.head(current_size)
#             tree = self.build_tree(training_set, algorithm)
#
#             correctness = self.evaluate_one_size(tree, test.head(1000))
#
#             val = {'training_size': current_size, 'correctness': correctness}
#             print(val)
#             result.append(val, ignore_index=True)
#
#         return result
#
#
# ###########################################################################################
# ###########################################################################################
# #########################From here on its testing only#####################################
# ##################################Ignore this##############################################
# ###########################################################################################
#
# data = dataset.get_data_for_model('../data/construction-data-processed.csv', balanced=False)
# cols_to_use = ['cat','arecon_cat_1000','matpar_3','matpis_8','matpar_7','arecon_cat_1000','numpis_cat_1+','matpis_22','numapo_cat_4']
# cols_to_remove = [i for i in list(data.columns.values) if i not in cols_to_use]
# data = data.drop(cols_to_remove, axis=1)
#
# #data = pd.read_csv('data.csv')
#
# builder = TreeBuilder('cat', 'entropy')
# evaluator = TreeEvaluator(builder)
# train, test = train_test_split(data, test_size=0.1)
# res = evaluator.evaluate(train, test, algorithm='cart')
#
# print(res)
#
