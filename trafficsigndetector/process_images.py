from preprocessor import *
from utils import resize_many


def prepare_for_classification(objects, image):
    images = []
    for obj in objects:
        x1, x2, y1, y2 = obj[1:]
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        img_cropped = image[y1:y2, x1:x2]
        images.append(img_cropped)
    images = resize_many(np.array(images), (40, 40))

    img_simple = [img for img in images]
    img_stretch = [preprocess(np.array([img]), histogram_stretching)[0] for img in images]
    img_eq = [preprocess(np.array([img]), histogram_equalization)[0] for img in images]
    img_adeq = [preprocess(np.array([img]), adaptive_histogram_equalization)[0] for img in images]
    img_hog = [hog(img) for img in images]

    preprocessed_images = [
        np.array([img_simple[i], img_stretch[i], img_eq[i], img_adeq[i], img_hog[i]])
        for i in range(0, len(img_simple))
    ]
    return np.array(preprocessed_images)
