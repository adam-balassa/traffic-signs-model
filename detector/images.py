import numpy as np

from preprocessor import preprocess, histogram_equalization, histogram_stretching, adaptive_histogram_equalization


def get_all_images(image, boxes):
    images = []
    for box in boxes:
        x1, x2, y1, y2 = box
        cropped_image = image[y1:y2, x1:x2]
        images.append(cropped_image)
    return np.array(images)


def preprocess_images(image, boxes):
    images = [
        preprocess([image], histogram_stretching)[0],
        preprocess([image], histogram_equalization)[0],
        preprocess([image], adaptive_histogram_equalization)[0]
    ]
    inputs = []
    for box_type in boxes:
        inputs.append([get_all_images(image, box_type) for image in images])
        inputs[-1].append(box_type)
    return inputs
