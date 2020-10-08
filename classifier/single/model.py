from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Input, Lambda, Add, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def sequential(layers):
    prev_layer = layers[0]
    for i in range(1, len(layers)):
        layer = layers[i]
        prev_layer = layer(prev_layer)
    return layers[0], prev_layer


def get_top_model(name):
    return sequential([
        Input(shape=(40, 40, 3), name=f'input_{name}'),
        Conv2D(32, kernel_size=3, padding='same', name=f'conv2d_top_{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f"max_pool_top_{name}")
    ])


def get_group_classifier_model(input_layer, name):
    return sequential([
        input_layer,
        Conv2D(16, kernel_size=3, padding='same', name=f'conv2d_group_cls_{name}', activation='relu'),
        Flatten(name=f'flatten_cls_{name}'),
        Dense(7, activation='softmax', name=f'group_cls_{name}')
    ])


def get_main_filter_map(input_layer, name):
    return sequential([
        input_layer,
        Conv2D(32, kernel_size=3, padding='same', name=f'conv2d_main1_{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f"max_pool_main1_{name}"),
        Conv2D(24, kernel_size=3, padding='same', name=f'conv2d_main2_{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f"max_pool_main2_{name}"),
        Flatten(name=f'flatten_{name}')
    ])


def linear_output(output, name):
    return Lambda(lambda x: x, name=name)(output)


def get_model(name):
    input_layer, top_output = get_top_model(name)
    cls_top, cls_output = get_group_classifier_model(top_output, name)
    main_map_top, main_map_output = get_main_filter_map(top_output, name)

    cls_output_layer = linear_output(cls_output, name=f'cls_output_{name}')

    _, main_output = sequential([
        [cls_output, main_map_output],
        Concatenate(name=f'concat_{name}'),
        Dense(300, activation='relu', name=f'dense_{name}'),
        Dense(43, activation='softmax', name=f'main_output_{name}')
    ])

    outputs = [cls_output_layer, main_output]
    return input_layer, outputs


def get_full_model():
    model_types = 'simple', 'stretch', 'eq', 'adeq'
    inputs, cls_outputs, main_outputs = [], [], []
    for model_type in model_types:
        input_layer, output_layers = get_model(model_type)
        inputs.append(input_layer)
        cls_out, main_out = output_layers
        cls_outputs.append(cls_out)
        main_outputs.append(main_out)

    _, common_cls = sequential([
        cls_outputs,
        Add(name='common_cls_add'),
        Lambda(lambda x: x / len(cls_outputs), name='common_cls_output')
    ])

    _, common_main = sequential([
        main_outputs,
        Add(name='common_main_add'),
        Lambda(lambda x: K.argmax(x, axis=-1), name='common_main_output')
    ])

    final_outputs = [common_cls, common_main]
    model = Model(inputs=inputs, outputs=final_outputs)
    return model