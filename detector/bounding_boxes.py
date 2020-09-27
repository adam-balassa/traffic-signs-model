import numpy as np

from detector.constants import SLIDING_WINDOW_RATIO


def get_all_boxes(image):
    boxes = np.empty((0, 4), dtype='int')
    i = 1
    while i <= 8:
        boxes = np.concatenate((boxes, get_boxes(image, i)), axis=0)
        i *= 2
    return boxes


def get_boxes(image, i):
    boxes = []
    box_size = int(image.shape[0] / i)
    dx = dy = box_size / SLIDING_WINDOW_RATIO
    number_of_boxes = int(SLIDING_WINDOW_RATIO * i - 1)
    x, y = 0, 0
    for i in range(0, number_of_boxes):
        for j in range(0, number_of_boxes):
            boxes.append([x, x + box_size, y, y + box_size])
            x += dx
        y += dy
        x = 0
    return np.array(boxes, dtype='int')
