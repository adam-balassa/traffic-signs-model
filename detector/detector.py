import numpy as np

from detector.bounding_boxes import scan_boxes, get_cropped_images, get_all_boxes
from detector.constants import DETECTOR_MODELS
from detector.load_models import load
from detector.non_max_suppression import non_max_suppression
from utils import time


class ObjectDetector:
    def __init__(self):
        self.cnns = load(DETECTOR_MODELS)

    def predict(self, image_eq, image_stretch, image_adeq):
        boxes = time.measure(lambda: get_all_boxes(image_eq), 'counting boxes')
        cropped_images = time.measure(lambda: get_cropped_images(boxes, image_eq, image_stretch, image_adeq), 'cropping windows')

        predictions = time.measure(lambda: scan_boxes(self.cnns, cropped_images, boxes), 'localization')
        result = non_max_suppression(predictions)

        return np.array(result)
