import numpy as np
from cv2 import resize

from preprocessor import preprocess, histogram_equalization, histogram_stretching, adaptive_histogram_equalization
from utils import time


def get_cropped_images(boxes, image):
    resized_images = []
    for box in boxes:
        x1, x2, y1, y2 = box
        cropped = image[y1:y2, x1:x2]
        resized_images.append(resize(cropped, (48, 48)))
    return np.array(resized_images)


def preprocess_images(image, boxes):
    images = get_cropped_images(boxes, image)
    adeq_images = preprocess(np.array([image]), adaptive_histogram_equalization)[0]
    preprocessed_images = preprocess(images, histogram_equalization), \
            preprocess(images, histogram_stretching), \
            get_cropped_images(boxes, adeq_images)
    return np.array(preprocessed_images)
