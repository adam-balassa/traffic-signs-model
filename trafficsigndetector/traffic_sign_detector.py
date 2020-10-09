from detector import ObjectDetector
from classifier.single import Classifier
from utils import time
from .bounding_box import extend_bounding_boxes
from .process_images import prepare_for_classification, prepare_multiple_for_classification
import numpy as np


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

    def detect_multiple(self, images):
        objects = time.measure(lambda: self.detector.predict_multiple(images), 'detection')
        [extend_bounding_boxes(obj, 0.15) for obj in objects]
        preprocessed = time.measure(lambda: prepare_multiple_for_classification(objects, images), 'image preprocessing')
        labels = time.measure(lambda: self.classifier.predict(preprocessed), 'classification')

        results = []
        j = 0
        for i in range(0, len(images)):
            results.append([labels[k] for k in range(j, j + len(objects[i]))])
            j += len(objects[i])

        print(objects, results)
        return objects
