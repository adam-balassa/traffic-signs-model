from preprocessor import *
from utils import resize_many


def prepare_for_classification(objects, image):
    images = []
    for obj in objects:
        x1, x2, y1, y2 = obj[1:]
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        img_cropped = image[y1:y2, x1:x2]
        images.append(img_cropped)
    images = resize_many(images, (40, 40))

    img_adeq = preprocess(images, adaptive_histogram_equalization)
    img_eq = preprocess(images, histogram_equalization)
    img_stretch = preprocess(images, histogram_stretching)

    return [images, img_stretch, img_eq, img_adeq]
