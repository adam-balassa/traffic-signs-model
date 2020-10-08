from detector.bounding_boxes import get_all_boxes
from detector.old.images import get_input
from detector.old.model import load_full_model
from detector.non_max_suppression import non_max_suppression
from utils import time


class ObjectDetector:
    def __init__(self):
        self.model = load_full_model()

    def predict(self, image):
        boxes = get_all_boxes(image)
        images = time.measure(lambda: get_input(image, boxes), 'image preprocessing')
        result = time.measure(lambda: self.model.predict(images), 'localization')
        result = non_max_suppression(result)
        return result
