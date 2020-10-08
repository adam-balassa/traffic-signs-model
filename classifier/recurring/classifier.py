from classifier.recurring.load_model import load_model


class Classifier:
    def __init__(self):
        self.model = load_model()

    def predict(self, images):
        return self.model.predict(images)
