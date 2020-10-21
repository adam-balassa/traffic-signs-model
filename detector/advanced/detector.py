import numpy as np

from detector.advanced.helpers import get_all_boxes, get_all_inputs, non_max_suppression
from detector.advanced.model import get_full_model
from utils import time


class ObjectDetector:
    def __init__(self):
        self.models = [get_full_model(n) for n in (256, 128, 64, 32)]

    def predict_multiple(self, images):
        boxes = get_all_boxes()
        box_scales = [len(image) / 256 for image in images]
        inputs, preprocessed_images = time.measure(lambda: get_all_inputs(images, boxes, box_scales), 'preprocess')
        results = None
        for i in range(0, 4):
            cls, reg = time.measure(lambda: self.models[i].predict(inputs[i]), f'detection {i}')
            cls, reg = np.asarray(cls), np.asarray(reg)
            result = np.reshape(
                np.concatenate((cls[..., 1:], reg), axis=-1),
                (len(images), len(boxes[i]), 5)
            )
            if results is None:
                results = result
            else:
                results = np.concatenate((results, result), axis=1)
        results = [non_max_suppression(result) for result in results]
        return results, preprocessed_images
