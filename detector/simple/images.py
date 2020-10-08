import numpy as np
from cv2 import resize

from preprocessor import preprocess, histogram_equalization, histogram_stretching, adaptive_histogram_equalization, \
    normalize


def get_cropped_images(boxes, image):
    resized_images = []
    for box in boxes:
        x1, x2, y1, y2 = box
        cropped = image[y1:y2, x1:x2]
        resized_images.append(resize(cropped, (48, 48)))
    return np.array(resized_images)


def get_input(image, boxes):
    images = get_cropped_images(boxes, image)
    preprocessed_images = [
        normalize(images),
        preprocess(images, histogram_stretching),
        preprocess(images, histogram_equalization),
        boxes
    ]
    return preprocessed_images
