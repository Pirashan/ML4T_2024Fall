import numpy as np
import LinRegLearner as lrl
import BagLearner as bl

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = [bl.BagLearner(lrl.LinRegLearner, {}, 20) for _ in range(20)]
        self.verbose = verbose
    def author(self):
        return ('pravikumaran3')
    def add_evidence(self, Xtrain, Ytrain):
        for learner in self.learners:
            learner.add_evidence(Xtrain, Ytrain)
    def query(self, Xtest):
        predictions = np.array([learner.query(Xtest) for learner in self.learners])
        return np.mean(predictions, axis=0)