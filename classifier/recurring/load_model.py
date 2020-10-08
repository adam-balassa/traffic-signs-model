from pathlib import Path

from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

import classifier
from classifier.recurring.model import get_full_model, model_names, get_cnn_model, get_model, get_release_model, \
    recurring_layer_multiplexer, get_model_simple


def load_model():
    inputs = [(Input(shape=(40, 40, 3), name=f'input_{name}')) for name in model_names]

    multilayer = get_full_model('multilayer', inputs, get_cnn_model)

    final_layers = {
        "prohibitory": get_full_model('prohibitory', inputs, lambda name, input_layer: get_model(name, input_layer, 13)),
        "danger": get_full_model('danger', inputs, lambda name, input_layer: get_model(name, input_layer, 16)),
        "direction": get_full_model('direction', inputs, lambda name, input_layer: get_model(name, input_layer, 9)),
        "release": get_release_model(inputs),
        "red_surface": get_full_model('red_surface', inputs, lambda name, input_layer: get_model_simple(name, input_layer, 3))
    }

    outputs = [multilayer, *[final_layers[name] for name in final_layers]]
    output = Lambda(lambda x: recurring_layer_multiplexer(x), name='layer_multiplexer')(outputs)

    Model(inputs=inputs, outputs=multilayer)\
        .load_weights(f'{Path(classifier.__file__).parent}/models/multilayer_classifier.h5')

    for name in final_layers:
        layer = final_layers[name]
        if name == 'release':
            filename = 'release_adeq'
        else:
            filename = f'{name}_classifier'
        Model(inputs=inputs, outputs=layer)\
            .load_weights(f'{Path(classifier.__file__).parent}/models/{filename}.h5')

    model = Model(inputs=inputs, outputs=output)
    return model
