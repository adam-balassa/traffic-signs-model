from classifier.classifier import TrafficSignClassifier
from .bounding_box import extend_bounding_boxes
from .process_images import prepare_for_classification
from detector import ObjectDetector
from utils import time


class TrafficSignDetector(object):
    def __init__(self):
        self.detector = ObjectDetector()
        self.classifier = TrafficSignClassifier()

    def detect(self, image):
        objects = time.measure(lambda: self.detector.predict(image), 'detection')
        extend_bounding_boxes(objects, 0.15)
        images = time.measure(lambda: prepare_for_classification(objects, image), 'image preprocessing')
        labels = time.measure(lambda: self.classifier.predict(images), 'classification')
        print(objects, labels)
        return objects, labels
