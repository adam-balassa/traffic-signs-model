import numpy as np
from cv2 import resize


def resize_many(images, shape=(32, 32)):
    return np.array([resize(img, shape) for img in images])
