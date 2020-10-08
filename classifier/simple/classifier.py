from classifier.simple.load_models import load_model

from classifier.simple.model import extend_model


class Classifier:
    def __init__(self):
        cnns = load_model()
        self.model = extend_model(cnns)
        self.model.compile()

    def predict(self, x):
        return self.model.predict(x)

