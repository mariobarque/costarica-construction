from tree.leaf import Leaf


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + 'Prediction: ' + node.print())
        return

    print(spacing + str(node.feature) + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print(spacing + str(node.feature) + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def predict_rec(row, node):
    if isinstance(node, Leaf):
        return node.value

    if row[node.feature] == 0:
        return predict_rec(row, node.false_branch)
    else:
        return predict_rec(row, node.true_branch)


def predict(df, node):
    predictions = []
    for _, row in df.iterrows():
        prediction = predict_rec(row, node)
        predictions.append(prediction)

    return predictions