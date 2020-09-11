from server import json
from trafficsigndetector.traffic_sign_detector import TrafficSignDetector
from utils import load_image, time


class Application:
    def __init__(self):
        self.detector = TrafficSignDetector()

    def detect(self, path):
        image = load_image(path)
        objects, labels = time.measure(lambda: self.detector.detect(image), 'the whole process')
        return json.convert(objects, labels)


