import numpy as np

from preprocessor.normalization import normalize


def preprocess(imgs, equalization):
    imgs = np.array([equalization(img) for img in imgs])
    imgs = normalize(imgs)
    return np.array(imgs)
