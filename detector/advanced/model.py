from pathlib import Path

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, AveragePooling2D, Dense, Input, Lambda, Add
from tensorflow.keras.models import Model

import detector


def sequential(layers):
    prev_layer = layers[0]
    for i in range(1, len(layers)):
        layer = layers[i]
        prev_layer = layer(prev_layer)
    return layers[0], prev_layer


def get_top_model(name, size):
    if size == 256:
        n_pool = 3
    elif size == 128:
        n_pool = 2
    elif size == 64:
        n_pool = 1
    else:
        n_pool = 0
    return sequential([
        Input(shape=(size, size, 3), name=f'input_{name}'),
        *[AveragePooling2D(2, name=f'norm_pooling_{name}_{i}') for i in range(0, n_pool)],
        Conv2D(32, kernel_size=3, padding='same', name=f'conv2d_top1_{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f"max_pool_top_{name}"),
        Conv2D(32, kernel_size=3, padding='same', name=f'conv2d_top2_{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f"max_pool_top2_{name}")
    ])


def get_classifier_model(input_layer, name):
    return sequential([
        input_layer,
        Flatten(name=f'flatten_cls_{name}'),
        Dense(2, activation='softmax', name=f'cls_{name}')
    ])


def get_box_regression_model(input_layer, name):
    return sequential([
        input_layer,
        Flatten(name=f'flatten_reg_{name}'),
        Dense(400, activation='relu', name=f'dense_reg_{name}'),
        Dense(4, activation='relu', name=f'reg_{name}')
    ])


def linear_output(output, name):
    return Lambda(lambda x: x, name=name)(output)


def get_model(name, size):
    input_layer, top_output = get_top_model(name, size)
    _, cls_output = get_classifier_model(top_output, name)
    _, reg_output = get_box_regression_model(top_output, name)

    outputs = [cls_output, reg_output]
    return input_layer, outputs


def normalize(x):
    box, bbox = x

    x1, x2, y1, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    box_sizes = x2 - x1

    x, y, w, h = bbox[..., 2], bbox[..., 3], bbox[..., 0], bbox[..., 1]
    bx, by, bw, bh = x * box_sizes + x1, y * box_sizes + y1, w * box_sizes, h * box_sizes

    predictions = [bx, by, bw, bh]
    return K.transpose(predictions)


def get_full_model(size):
    model_types = 'simple', 'stretch', 'eq'
    inputs, reg_outputs, cls_outputs = [], [], []
    for model_type in model_types:
        input_layer, output_layers = get_model(model_type, size)
        inputs.append(input_layer)
        cls_output, reg_output = output_layers
        reg_outputs.append(reg_output)
        cls_outputs.append(cls_output)

    _, common_cls = sequential([
        cls_outputs,
        Add(name='cls_add'),
        Lambda(lambda x: x / len(cls_outputs), name='common_cls')
    ])
    _, common_reg = sequential([
        reg_outputs,
        Add(name='reg_add'),
        Lambda(lambda x: x / len(cls_outputs), name='common_reg')
    ])

    Model(inputs=inputs, outputs=[common_cls, common_reg])\
        .load_weights(f'{Path(detector.__file__).parent}/models/simple_detector_3.h5')

    box_shape_input = Input(shape=[4], name="box_shape")
    inputs.append(box_shape_input)
    normalize_layer = Lambda(lambda x: normalize(x), name="normalize")([box_shape_input, common_reg])

    return Model(inputs=inputs, outputs=[common_cls, normalize_layer])
