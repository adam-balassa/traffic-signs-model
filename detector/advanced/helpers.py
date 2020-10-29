from skimage.transform import resize
import numpy as np

from detector.constants import CLASSIFICATION_THRESHOLD, SLIDING_WINDOW_RATIO
from preprocessor import normalize, preprocess, histogram_stretching, histogram_equalization


def get_cropped_images(images, boxes):
    cropped = []
    for image in images:
        for box in boxes:
            x1, x2, y1, y2 = box
            img = image[y1:y2, x1:x2]
            cropped.append(img)
    return np.asarray(cropped)


def get_input(image, boxes):
    images = get_cropped_images(boxes, image)
    preprocessed_images = [
        normalize(images),
        preprocess(images, histogram_stretching),
        preprocess(images, histogram_equalization),
        boxes
    ]
    return preprocessed_images


def get_preprocessed_images(images):
    images = [
        normalize(images),
        preprocess(images, histogram_stretching),
        preprocess(images, histogram_equalization)
    ]
    images = [np.array([resize(img, (256, 256)) for img in imgs]) for imgs in images]
    return images


def get_inputs(images, boxes):
    n = len(images[0])
    images = [get_cropped_images(imgs, boxes) for imgs in images]
    return [*images, boxes]


def get_all_inputs(images, boxes):
    images = get_preprocessed_images(images)
    inputs = []
    for box_type in boxes:
        inputs.append(get_inputs(images, box_type))
    return inputs, images


def get_all_boxes():
    boxes = []
    i = 1
    while i <= 8:
        boxes.append(get_boxes(i))
        i *= 2
    return boxes


def get_boxes(i):
    boxes = []
    box_size = int(256 / i)
    dx = dy = box_size / SLIDING_WINDOW_RATIO
    number_of_boxes = int((i - 1) * SLIDING_WINDOW_RATIO + 1)
    x, y = 0, 0
    for i in range(0, number_of_boxes):
        for j in range(0, number_of_boxes):
            x_m, y_m = min(x + box_size, 256), min(y + box_size, 256)
            boxes.append([x_m - box_size, x_m, y_m - box_size, y_m])
            x += dx
        y += dy
        x = 0
    return np.array(boxes, dtype='int')


def iou(ax, ay, aw, ah, bx, by, bw, bh):
    ax1, ax2, ay1, ay2 = ax - aw / 2, ax + aw / 2, ay - ah / 2, ay + ah / 2
    bx1, bx2, by1, by2 = bx - bw / 2, bx + bw / 2, by - bh / 2, by + bh / 2
    x1, x2 = max(ax1, bx1), min(ax2, bx2)
    y1, y2 = max(ay1, by1), min(ay2, by2)
    intersection = (x2 - x1) * (y2 - y1)
    area_a, area_b = (ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)
    return max(intersection / area_a, intersection / area_b)


def find_high_ious(predictions, box):
    matches = []
    ax1, ax2, ay1, ay2 = box[1:]
    for pred in predictions:
        bx1, bx2, by1, by2 = pred[1:]
        iou_score = iou(ax1, ax2, ay1, ay2, bx1, bx2, by1, by2)
        if iou_score > 0.38:
            matches.append(True)
        elif iou_score > 0.05 and ((ax2 - ax1) / (bx2 - bx1)) > 1.3:
            matches.append(True)
        else:
            matches.append(False)
    return np.array(matches)


def non_max_suppression(predictions):
    predictions = predictions[predictions[..., 0] > CLASSIFICATION_THRESHOLD]
    results = []
    while len(predictions) > 0:
        best_match = predictions[np.argmax(predictions[..., 0])]
        mask = find_high_ious(predictions, best_match)
        predictions = predictions[~mask]
        results.append(best_match)
    return np.array(results)