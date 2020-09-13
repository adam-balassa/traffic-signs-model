from classifier.constants import CLASSIFIER_MODELS
from classifier.load_models import load_model
import numpy as np


class TrafficSignClassifier:
    def __init__(self):
        self.cnns = load_model()

    def predict(self, images):
        results = np.array([self.cnns[i].predict(np.array([images[i]]))[0] for i in range(0, len(self.cnns))])
        return self.vote(results)

    @staticmethod
    def vote(predictions):
        votes = [np.argmax(pred) for pred in predictions[:-1]]
        votes.append(predictions[2])
        winner = np.argmax(np.bincount(votes, minlength=43))
        return winner
