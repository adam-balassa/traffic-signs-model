import numpy as np

from detector.bounding_boxes import get_boxes, scan_boxes
from detector.constants import DETECTOR_MODELS
from detector.load_models import load
from detector.non_max_suppression import non_max_suppression


class ObjectDetector:
    def __init__(self):
        self.cnns = load(DETECTOR_MODELS)

    def predict(self, image_eq, image_stretch, image_adeq):
        boxes = np.empty((0, 4), dtype='int')
        images = [image_eq, image_stretch, image_adeq]
        i = 1
        while i <= 8:
            boxes = np.concatenate((boxes, get_boxes(images[0], i)), axis=0)
            i *= 2
        predictions = scan_boxes(self.cnns, images, boxes)
        result = non_max_suppression(predictions)

        return np.array(result)
