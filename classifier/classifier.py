from classifier.constants import CLASSIFIER_MODELS
from classifier.load_models import load
import numpy as np


class TrafficSignClassifier:
    def __init__(self):
        self.cnns = load(CLASSIFIER_MODELS)

    def predict(self, images):
        results = np.array([self.cnns[i].predict(np.array([images[i]]))[0] for i in range(0, len(self.cnns))])
        return self.vote(results)

    @staticmethod
    def vote(predictions):
        votes = np.argmax(predictions, axis=1)
        winner = np.argmax(np.bincount(votes, minlength=43))
        return winner
