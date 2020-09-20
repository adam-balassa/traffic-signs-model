from pathlib import Path

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Input, Lambda

import detector
from detector.constants import DETECTOR_MODELS


def get_model(name):
    model = Sequential()
    model.add(
        Conv2D(48, kernel_size=4, padding='same', activation='relu', input_shape=(48, 48, 3), name=f"conv1_{name}"))
    model.add(MaxPool2D(pool_size=2, name=f"pool1_{name}"))
    model.add(Conv2D(24, kernel_size=4, padding='same', activation='relu', name=f"conv2_{name}"))
    model.add(MaxPool2D(pool_size=2, name=f"pool2_{name}"))
    model.add(Flatten(name=f"flatten_{name}"))
    model.add(Dropout(0.2, name=f"dropout_{name}"))
    model.add(Dense(300, activation='relu', name=f"dense_{name}"))
    model.add(Dense(5, activation='sigmoid', name=f"core_out_{name}"))
    load_weights(model, name)
    return model


def load_weights(model, name):
    model.load_weights('{path}/models/{filename}_{name}.h5'
                       .format(path=Path(detector.__file__).parent, filename=DETECTOR_MODELS, name=name))


def sequential(layers):
    prev_layer = layers[0]
    for i in range(1, len(layers)):
        layer = layers[i]
        prev_layer = layer(prev_layer)
    return layers[0], prev_layer


def summarize(x):
    x = K.permute_dimensions(tf.convert_to_tensor(x), (1, 0, 2))
    confidences, bounding_boxes = x[..., 0], x[..., 1:]

    positive_pred_mask = K.greater(confidences, 0.000005)
    positive_pred_mask = K.cast(positive_pred_mask, 'float32')

    sum_confidences = K.sum(confidences, axis=-1)
    sum_positive_pred_mask = K.cast(K.greater(sum_confidences, 0.7), 'float32')

    n_positive_pred = K.sum(positive_pred_mask, axis=-1)

    positive_boxes = bounding_boxes * K.expand_dims(positive_pred_mask, axis=-1)
    denominator = n_positive_pred + (1 - K.cast(K.greater(n_positive_pred, 0), 'float32'))
    avg_boxes = K.sum(positive_boxes, axis=-2) / K.expand_dims(denominator)

    boxes = avg_boxes * K.expand_dims(sum_positive_pred_mask)

    result = K.concatenate((K.expand_dims(sum_confidences / 3), boxes), axis=-1)

    return result


def normalize(x):
    box, pred = x
    confidences, bbox = pred[..., 0], pred[..., 1:]

    x1, x2, y1, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    box_sizes = x2 - x1

    x, y, w, h = bbox[..., 2], bbox[..., 3], bbox[..., 0], bbox[..., 1]
    bx, by, bw, bh = x * box_sizes + x1, y * box_sizes + y1, w * box_sizes, h * box_sizes
    bx1, bx2, by1, by2 = bx - bw / 2, bx + bw / 2, by - bh / 2, by + bh / 2

    predictions = [confidences, bx1, bx2, by1, by2]
    return K.transpose(predictions)


def load_full_model():
    names = ['stretch', 'eq', 'adeq']

    models = [get_model(name) for name in names]

    inputs, outputs = [], []
    for i in range(0, len(models)):
        input_layer, output = sequential([
            Input(shape=(48, 48, 3), name=f"input_{names[i]}"),
            *models[i].layers
        ])
        inputs.append(input_layer), outputs.append(output)

    box_shape_input = Input(shape=4, name="box_shape")
    inputs.append(box_shape_input)

    summarize_layer = Lambda(lambda x: summarize(x), name="summarize")(outputs)
    normalize_layer = Lambda(lambda x: normalize(x), name="normalize")([box_shape_input, summarize_layer])

    return Model(inputs=inputs, outputs=[normalize_layer])
