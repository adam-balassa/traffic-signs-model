import numpy as np

from detector.bounding_boxes import scan_boxes, get_all_boxes
from detector.constants import DETECTOR_MODELS
from detector.images import get_cropped_images, preprocess_images
from detector.load_models import load
from detector.non_max_suppression import non_max_suppression
from utils import time


class ObjectDetector:
    def __init__(self):
        self.cnns = load(DETECTOR_MODELS)

    def predict(self, image):
        boxes = time.measure(lambda: get_all_boxes(image), 'counting boxes')
        preprocessed_images = time.measure(lambda: preprocess_images(image, boxes), 'preprocess images')

        predictions = time.measure(lambda: scan_boxes(self.cnns, preprocessed_images, boxes), 'localization')
        result = non_max_suppression(predictions)

        return np.array(result)
