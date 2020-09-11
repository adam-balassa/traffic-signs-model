from classifier.classifier import TrafficSignClassifier
from .bounding_box import extend_bounding_boxes
from .process_images import prepare_for_detection, prepare_for_classification
from detector import ObjectDetector
from utils import time


class TrafficSignDetector(object):
    def __init__(self):
        self.detector = ObjectDetector()
        self.classifier = TrafficSignClassifier()

    def detect(self, image):
        imgEq, imgStretch, imgAdEq = time.measure(lambda: prepare_for_detection(image), 'image preprocessing')
        objects = time.measure(lambda: self.detector.predict(imgEq, imgStretch, imgAdEq), 'detection')
        extend_bounding_boxes(objects, 0.15)
        images = time.measure(lambda: prepare_for_classification(objects, image, imgEq, imgStretch, imgAdEq), 'image preprocessing')
        labels = time.measure(lambda: [self.classifier.predict(imgs) for imgs in images], 'classification')
        return objects, labels
