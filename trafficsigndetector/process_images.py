from skimage.transform import resize

from preprocessor import *
from utils import resize_many


def get_cropped_images(objects, image):
    images = []
    for obj in objects:
        x1, x2, y1, y2 = obj[1:]
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        img_cropped = image[y1:y2, x1:x2]
        images.append(img_cropped)
    return resize_many(images, (40, 40))


def prepare_for_classification(objects, image):
    images = get_cropped_images(objects, image)

    img_adeq = preprocess(images, adaptive_histogram_equalization)
    img_eq = preprocess(images, histogram_equalization)
    img_stretch = preprocess(images, histogram_stretching)

    return [images, img_stretch, img_eq, img_adeq]


def prepare_multiple_for_classification(objects, images):
    cropped_images = \
        np.array([imgs for i in range(0, len(objects)) for imgs in get_cropped_images(objects[i], images[i])])

    img_adeq = preprocess(cropped_images, adaptive_histogram_equalization)
    img_eq = preprocess(cropped_images, histogram_equalization)
    img_stretch = preprocess(cropped_images, histogram_stretching)

    return [cropped_images, img_stretch, img_eq, img_adeq]


def resize_for_classification(objects, preprocessed, original):
    simple_, stretch_, eq_ = preprocessed
    adeq_ = preprocess(original, adaptive_histogram_equalization)
    simple, stretch, eq, adeq = [], [], [], []
    for i in range(0, len(objects)):
        for x, y, w, h in objects[i][..., 1:]:
            x1, x2, y1, y2 = x-w/2, x+w/2, y-h/2, y+h/2
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            simple.append(resize(simple_[i, y1:y2, x1:x2], (40, 40)))
            stretch.append(resize(stretch_[i, y1:y2, x1:x2], (40, 40)))
            eq.append(resize(eq_[i, y1:y2, x1:x2], (40, 40)))
            adeq.append(resize(adeq_[i, y1:y2, x1:x2], (40, 40)))

    return np.asarray(simple), np.asarray(stretch), np.asarray(eq), np.asarray(adeq)
