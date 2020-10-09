from classifier.single.load_model import load_model
from classifier.single.model import get_full_model
import numpy as np


class Classifier:
    def __init__(self):
        self.model = get_full_model()
        load_model(self.model)

    def predict(self, images):
        return self.model.predict(images)[-1]
