from preprocessor import *
from utils.resize import resize_many


def prepare_for_detection(image):
    image = [image]
    images = preprocess(image, histogram_equalization), preprocess(image, histogram_stretching), preprocess(image, adaptive_histogram_equalization)
    return np.array(images)[:, 0]


def prepare_for_classification(objects, image, img_eq, img_stretch, img_adeq):
    images = []
    for obj in objects:
        x1, x2, y1, y2 = obj[1:]
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        img_simple = image[y1:y2, x1:x2]
        img_hog = hog(img_simple)
        resized = resize_many([img_simple, img_stretch[y1:y2, x1:x2], img_eq[y1:y2, x1:x2], img_adeq[y1:y2, x1:x2]],
                              (40, 40))
        resized[0] = normalize(np.array([resized[0]]))[0]
        img = [*resized, img_hog]
        images.append(img)
    return images
