from pathlib import Path

from joblib import load

import classifier
from classifier.model import get_model


def load_model():
    simple_model = get_model()
    simple_model.load_weights('{path}/models/model_committee0.h5'.format(path=Path(classifier.__file__).parent))

    adeq_model = get_model()
    adeq_model.load_weights('{path}/models/model_committee3.h5'.format(path=Path(classifier.__file__).parent))

    regressor = load('{path}/models/logistic_regressor.joblib'.format(path=Path(classifier.__file__).parent))
    return [simple_model, adeq_model, regressor]
