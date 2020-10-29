import numpy as np

from detector.constants import SLIDING_WINDOW_RATIO


def get_all_boxes(image):
    boxes = np.empty((0, 4), dtype='int')
    i = 1
    while i <= 8:
        boxes = np.concatenate((boxes, get_boxes(image, i)), axis=0)
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
