from pathlib import Path

import classifier
from classifier.model import get_model, get_mlp


def load(filename):
    cnns = []
    for i in range(0, 5):
        if i == 4:
            model = get_mlp()
        else:
            model = get_model()
        model.load_weights(
            '{path}/models/{filename}{name}.h5'.format(path=Path(classifier.__file__).parent, filename=filename, name=i))
        cnns.append(model)
    return cnns
