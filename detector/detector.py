import numpy as np

from detector.bounding_boxes import get_all_boxes
from detector.images import preprocess_images
from detector.model import load_full_model
from detector.non_max_suppression import non_max_suppression


class ObjectDetector:
    def __init__(self):
        self.model = load_full_model()

    def predict(self, image):
        boxes = get_all_boxes(image)
        image_types = preprocess_images(image, boxes)

        result = np.ndarray(shape=(0, 5), dtype='float32')
        for images in image_types:
            predictions = self.model.predict(images)
            result = np.concatenate((result, predictions))

        result = non_max_suppression(result)
        return result
