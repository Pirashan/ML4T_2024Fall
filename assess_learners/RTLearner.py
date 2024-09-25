import numpy as np
import random

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return('pravikumaran3')

    def study_group(self):
        return('pravikumaran3')

    def add_evidence(self, Xtrain, Ytrain):
        self.tree = self.build_tree(Xtrain, Ytrain)
        if self.verbose:
            print("tree built:", self.tree)

    def build_tree(self, dataX, dataY):
        if dataX.shape[0] <= self.leaf_size or np.all(dataY == dataY[0]):
            return np.array([[-1, np.mean(dataY), np.nan, np.nan]])

        best_feature = self.get_best_feature(dataX)
        split_val = np.median(dataX[:, best_feature])

        if split_val == np.max(dataX[:, best_feature]):
            return np.array([[-1, np.mean(dataY), np.nan, np.nan]])

        left_index = dataX[:, best_feature] <= split_val
        right_index = dataX[:, best_feature] > split_val

        if np.sum(right_index) == 0:
            split_val = np.mean(dataX[:, best_feature])
            right_index = dataX[:, best_feature] > split_val

        left_tree = self.build_tree(dataX[left_index], dataY[left_index])
        right_tree = self.build_tree(dataX[right_index], dataY[right_index])

        root = np.array([[best_feature, split_val, 1, len(left_tree) + 1]])

        return np.vstack((root, left_tree, right_tree))

    def get_best_feature(self, X):
        # Randomly select a feature to split on
        return random.randint(0, X.shape[1] - 1)

    def query(self, Xtest):
        predictions = np.apply_along_axis(self.query_point, 1, Xtest)
        return predictions

    def query_point(self, point):
        node = 0
        while self.tree[node, 0] != -1:  # while it's not a leaf
            feature_index = int(self.tree[node, 0])
            split_value = self.tree[node, 1]
            if point[feature_index] <= split_value:
                node += int(self.tree[node, 2])  # left child
            else:
                node += int(self.tree[node, 3])  # right child
        return self.tree[node, 1]