import numpy as np

from utils import resize_many


def get_all_boxes(images):
    boxes = np.empty((0, 4), dtype='int')
    i = 1
    while i <= 8:
        boxes = np.concatenate((boxes, get_boxes(images, i)), axis=0)
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


def get_predictions(cnns, images):
    predictions = np.array([cnns[i].predict(images[i]) for i in range(0, 3)])
    predictions = np.transpose(predictions, axes=(1, 0, 2))
    final_predictions = []
    for prediction in predictions:
        if np.sum(prediction[..., 0]) < 0.7:
            final_predictions.append([0, 0, 0, 0, 0])
        else:
            true_predictions = prediction[prediction[..., 0] > 0.5]
            pred = [np.sum(prediction[..., 0]) / 3, *(np.sum(true_predictions[..., 1:], axis=0) / len(true_predictions))]
            final_predictions.append(pred)
    return np.array(final_predictions)


def get_cropped_images(boxes, image_eq, image_stretch, image_adeq):
    images = [image_eq, image_stretch, image_adeq]
    resized_images = []
    for box in boxes:
        x1, x2, y1, y2 = box
        cropped = [image[y1:y2, x1:x2] for image in images]
        resized_images.append(resize_many(cropped, (48, 48)))
    resized_images = np.transpose(np.array(resized_images), axes=(1, 0, 2, 3, 4))
    return resized_images


def scan_boxes(cnns, resized_images, boxes):
    predictions = get_predictions(cnns, resized_images)

    for i in range(0, len(boxes)):
        x1, x2, y1, y2 = boxes[i]
        box_size = x2 - x1
        pred = predictions[i]
        x = box_size * pred[3] + x1
        y = box_size * pred[4] + y1
        w = box_size * pred[1]
        h = box_size * pred[2]
        pred[1] = x - w / 2
        pred[2] = x + w / 2
        pred[3] = y - h / 2
        pred[4] = y + h / 2
    return predictions
