from pathlib import Path

import detector


def load_model(model):
    model.load_weights(f'{Path(detector.__file__).parent}/models/simple_detector_2.h5')
