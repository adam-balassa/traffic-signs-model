from detector import ObjectDetector
from classifier.single import Classifier
from utils import time
from .bounding_box import extend_bounding_boxes
from .process_images import prepare_for_classification


class TrafficSignDetector(object):
    def __init__(self):
        self.detector = ObjectDetector()
        self.classifier = Classifier()

    def detect(self, image):
        objects = time.measure(lambda: self.detector.predict(image), 'detection')
        extend_bounding_boxes(objects, 0.15)
        images = time.measure(lambda: prepare_for_classification(objects, image), 'image preprocessing')
        labels = time.measure(lambda: self.classifier.predict(images), 'classification')
        print(objects, labels)
        return objects, labels
