import numpy as np


def get_all_boxes(image):
    boxes = []
    i = 1
    while i <= 8:
        boxes.append(get_boxes(image, i))
        i *= 2
    return boxes


def get_boxes(image, i):
    boxes = []
    box_size = int(image.shape[0] / i)
    number_of_boxes = 2 * i - 1
    x, y = 0, 0
    for i in range(0, number_of_boxes):
        for j in range(0, number_of_boxes):
            boxes.append([x, x + box_size, y, y + box_size])
            x += box_size / 2
        y += box_size / 2
        x = 0
    return np.array(boxes, dtype='int')