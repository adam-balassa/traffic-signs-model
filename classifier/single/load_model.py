from pathlib import Path

import classifier


def load_model(model):
    model.load_weights(f'{Path(classifier.__file__).parent}/models/single_multilayer_classifier.h5')
