import numpy as np


def iou(ax1, ax2, ay1, ay2, bx1, bx2, by1, by2):
    x1, x2 = max(ax1, bx1), min(ax2, bx2)
    y1, y2 = max(ay1, by1), min(ay2, by2)
    intersection = (x2 - x1) * (y2 - y1)
    area_a, area_b = (ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)
    return max([intersection / area_a, intersection / area_b])


def find_high_ious(predictions, box):
    matches = []
    ax1, ax2, ay1, ay2 = box[1:]
    for pred in predictions:
        bx1, bx2, by1, by2 = pred[1:]
        if iou(ax1, ax2, ay1, ay2, bx1, bx2, by1, by2) > 0.33:
            matches.append(True)
        else:
            matches.append(False)
    return np.array(matches)


def non_max_suppression(predictions):
    predictions = predictions[predictions[..., 0] > 0.8]
    results = []
    while len(predictions) > 0:
        best_match = predictions[np.argmax(predictions[..., 0])]
        mask = find_high_ious(predictions, best_match)
        predictions = predictions[~mask]
        results.append(best_match)
    return np.array(results)