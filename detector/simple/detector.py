import numpy as np

from detector.bounding_boxes import get_all_boxes
from detector.simple.images import get_input
from detector.non_max_suppression import non_max_suppression
from detector.simple.model import get_full_model
from utils import time


class ObjectDetector:
    def __init__(self):
        self.model = get_full_model()

    def predict(self, image):
        boxes = get_all_boxes(image)
        images = time.measure(lambda: get_input(image, boxes), 'image preprocessing')
        cls, reg = time.measure(lambda: self.model.predict(images), 'localization')
        result = np.concatenate((cls[..., 1:], reg), axis=-1)
        result = non_max_suppression(result)
        return result
