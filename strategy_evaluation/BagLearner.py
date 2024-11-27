import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))
        self.bags = bags
        self.verbose = verbose

    def author(self):
        return ('pravikumaran3')

    def study_group(self):
        return ('pravikumaran3')

    def add_evidence(self, Xtrain, Ytrain):
        n = Xtrain.shape[0]
        for i in range(self.bags):
            indices = np.random.choice(n, n, replace=True)
            X_sample = Xtrain[indices]
            Y_sample = Ytrain[indices]

            self.learners[i].add_evidence(X_sample, Y_sample)

    def query(self, points):
        predictions = np.array([learner.query(points) for learner in self.learners])
        return np.mean(predictions, axis=0)