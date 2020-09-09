from server import json
from trafficsigndetector.traffic_sign_detector import TrafficSignDetector
from utils.load_image import load_image


class Application:
    def __init__(self):
        self.detector = TrafficSignDetector()

    def detect(self, path):
        image = load_image(path)
        objects, labels = self.detector.detect(image)
        return json.convert(objects, labels)


