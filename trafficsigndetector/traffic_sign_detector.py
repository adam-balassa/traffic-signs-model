from classifier.single import Classifier
from detector import ObjectDetector
from utils import time
from .bounding_box import extend_bounding_boxes
from .process_images import prepare_for_classification, resize_for_classification


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
        objects, preprocessed_images, box_scales = time.measure(lambda: self.detector.predict_multiple(images), 'detection')
        if len(objects[0]) == 0:
            return [[]], [[]]
        preprocessed = time.measure(lambda: resize_for_classification(objects, preprocessed_images, images), 'preprocessing')
        labels = time.measure(lambda: self.classifier.predict(preprocessed), 'classification')

        results = []
        j = 0
        for i in range(0, len(images)):
            results.append([labels[k] for k in range(j, j + len(objects[i]))])
            j += len(objects[i])

        r_objects = []
        for i in range(0, len(objects)):
            r_objects.append([])
            objs = objects[i]
            r_objects[i] = [[obj[0], *obj[1:] * box_scales[i]] for obj in objs]
        print(r_objects, results)
        return r_objects, results
