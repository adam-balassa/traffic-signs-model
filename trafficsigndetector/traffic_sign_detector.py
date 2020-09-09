from classifier.classifier import TrafficSignClassifier
from .bounding_box import extend_bounding_boxes
from .process_images import prepare_for_detection, prepare_for_classification
from detector import ObjectDetector


class TrafficSignDetector(object):
    def __init__(self):
        self.detector = ObjectDetector()
        self.classifier = TrafficSignClassifier()

    def detect(self, image):
        imgEq, imgStretch, imgAdEq = prepare_for_detection(image)
        objects = self.detector.predict(imgEq, imgStretch, imgAdEq)
        extend_bounding_boxes(objects, 0.15)
        images = prepare_for_classification(objects, image, imgEq, imgStretch, imgAdEq)
        labels = [self.classifier.predict(imgs) for imgs in images]
        return objects, labels
