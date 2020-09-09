def extend_bounding_boxes(objects, ratio):
    for obj in objects:
        x1, x2, y1, y2 = obj[1:]
        w, h = x2 - x1, y2 - y1
        dx, dy = w * ratio / 2, h * ratio / 2
        x1, x2, y1, y2 = x1 - dy, x2 + dx, y1 - dy, y2 + dy
        obj[1], obj[2], obj[3], obj[4] = x1, x2, y1, y2
