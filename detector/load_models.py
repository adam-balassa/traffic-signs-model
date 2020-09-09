from pathlib import Path

import detector
from detector.model import get_model

names = ['eq', 'stretch', 'adeq']


def load(filename):
    cnns = []
    for name in names:
        model = get_model()
        model.load_weights(
            '{path}/models/{filename}_{name}.h5'.format(path=Path(detector.__file__).parent, filename=filename, name=name))
        cnns.append(model)
    return cnns
