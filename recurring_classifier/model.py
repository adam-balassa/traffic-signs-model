from pathlib import Path

from joblib import load
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Input, Activation, Lambda, Add

import recurring_classifier


def sequential(layers):
    prev_layer = layers[0]
    for i in range(1, len(layers)):
        layer = layers[i]
        prev_layer = layer(prev_layer)
    return layers[0], prev_layer


def get_model(name, output):
    return sequential([
        Input(shape=(40, 40, 3), name=f'input_{name}'),
        Conv2D(16, kernel_size=3, padding='same', name=f'conv2d1_1{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f'maxpool_1{name}'),
        Conv2D(24, kernel_size=3, padding='same', name=f'conv2d1_2{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f'maxpool_2{name}'),
        Flatten(name=f'flatten_{name}'),
        Dense(300, name=f'dense1_{name}', activation='relu'),
        Dense(output, name=f'dense2_{name}', activation='softmax')
    ])


def get_cnn_model(name):
    return sequential([
        Input(shape=(40, 40, 3), name=f'input_{name}'),
        Conv2D(16, kernel_size=3, padding='same', name=f'conv2d1_1{name}'),
        Activation('relu', name=f'relu1_1{name}'),
        MaxPool2D(pool_size=2, name=f'maxpool_1{name}'),
        Conv2D(24, kernel_size=3, padding='same', name=f'conv2d1_2{name}'),
        Activation('relu', name=f'relu1_2{name}'),
        MaxPool2D(pool_size=2, name=f'maxpool_2{name}'),
        Flatten(name=f'flatten_{name}'),
        Dense(7, name=f'dense_{name}', activation='softmax')
    ])


def get_model_simple(name, output):
    return sequential([
        Input(shape=(40, 40, 3), name=f'input_{name}'),
        Conv2D(12, kernel_size=3, padding='same', name=f'conv2d1_1{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f'maxpool_1{name}'),
        Conv2D(16, kernel_size=3, padding='same', name=f'conv2d1_2{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f'maxpool_2{name}'),
        Flatten(name=f'flatten_{name}'),
        Dense(200, name=f'dense1_{name}', activation='relu'),
        Dense(output, name=f'dense2_{name}', activation='softmax')
    ])


def get_full_model(names, get_model):
    inputs, model_outputs = [], []
    for name in names:
        model_input, model_output = get_model(name)

        inputs.append(model_input)
        model_outputs.append(model_output)

    _, main_output = sequential([
        Add(name='votes_sum')(model_outputs),
        Lambda(lambda x: x / len(names), name='main_output')
    ])

    model = Model(inputs=inputs, outputs=main_output)
    model.compile(optimizer='adam', metrics=['accuracy'])
    return model


def get_release_model():
    input_layer, output = get_model('model', 5)
    inputs = [Input(shape=(40, 40, 3)), Input(shape=(40, 40, 3)), Input(shape=(40, 40, 3)), input_layer]
    release_model = Model(inputs=inputs, outputs=output)
    release_model.compile(optimizer='adam', metrics=['accuracy'])
    return release_model


def get_multilayer_model():
    model_names = ('simple', 'stretch', 'eq', 'adeq')
    model = get_full_model(model_names, lambda name: get_cnn_model(name))
    model.load_weights('{path}/models/multilayer_classifier.h5'.format(path=Path(recurring_classifier.__file__).parent))
    return model


def get_final_layer_models():
    models = {}
    final_classes = {"prohibitory": 12, "danger": 15, "direction": 8, "release": 4, "red_surface": 2}
    for name in final_classes:
        output_size = final_classes[name]
        if name == 'release':
            models[name] = get_release_model()
            models[name].load_weights('{path}/models/release_adeq.h5'
                                      .format(path=Path(recurring_classifier.__file__).parent))
        elif output_size < 5:
            model_gen = lambda model_name: get_model_simple(model_name, output_size + 1)
            models[name] = get_full_model(['simple', 'stretch', 'eq', 'adeq'], model_gen)
            models[name].load_weights('{path}/models/{name}_classifier.h5'
                                      .format(path=Path(recurring_classifier.__file__).parent, name=name))
        else:
            model_gen = lambda model_name: get_model(model_name, output_size + 1)
            models[name] = get_full_model(['simple', 'stretch', 'eq', 'adeq'], model_gen)
            models[name].load_weights('{path}/models/{name}_classifier.h5'
                                      .format(path=Path(recurring_classifier.__file__).parent, name=name))
    return models


def get_y_value_map():
    return load('{path}/models/multilayer_y_values.joblib'.format(path=Path(recurring_classifier.__file__).parent))

