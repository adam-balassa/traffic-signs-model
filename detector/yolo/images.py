import numpy as np
from cv2 import resize

from preprocessor import normalize


def preprocess_images(images):
    return normalize(np.array([resize(image, (100, 100)) for image in images]))
