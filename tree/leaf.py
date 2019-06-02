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