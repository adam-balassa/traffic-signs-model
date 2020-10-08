import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Activation, Lambda, Add

model_names = 'simple', 'stretch', 'eq', 'adeq'


def sequential(layers):
    prev_layer = layers[0]
    for i in range(1, len(layers)):
        layer = layers[i]
        prev_layer = layer(prev_layer)
    return layers[0], prev_layer


def get_model(name, input_layer, output):
    return sequential([
        input_layer,
        Conv2D(16, kernel_size=3, padding='same', name=f'conv2d1_1{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f'maxpool_1{name}'),
        Conv2D(24, kernel_size=3, padding='same', name=f'conv2d1_2{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f'maxpool_2{name}'),
        Flatten(name=f'flatten_{name}'),
        Dense(300, name=f'dense1_{name}', activation='relu'),
        Dense(output, name=f'dense2_{name}', activation='softmax')
    ])


def get_cnn_model(name, input_layer):
    return sequential([
        input_layer,
        Conv2D(16, kernel_size=3, padding='same', name=f'conv2d1_1{name}'),
        Activation('relu', name=f'relu1_1{name}'),
        MaxPool2D(pool_size=2, name=f'maxpool_1{name}'),
        Conv2D(24, kernel_size=3, padding='same', name=f'conv2d1_2{name}'),
        Activation('relu', name=f'relu1_2{name}'),
        MaxPool2D(pool_size=2, name=f'maxpool_2{name}'),
        Flatten(name=f'flatten_{name}'),
        Dense(7, name=f'dense_{name}', activation='softmax')
    ])


def get_model_simple(name, input_layer, output):
    return sequential([
        input_layer,
        Conv2D(12, kernel_size=3, padding='same', name=f'conv2d1_1{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f'maxpool_1{name}'),
        Conv2D(16, kernel_size=3, padding='same', name=f'conv2d1_2{name}', activation='relu'),
        MaxPool2D(pool_size=2, name=f'maxpool_2{name}'),
        Flatten(name=f'flatten_{name}'),
        Dense(200, name=f'dense1_{name}', activation='relu'),
        Dense(output, name=f'dense2_{name}', activation='softmax')
    ])


def get_2_max(y):
    best = K.argmax(y, axis=-1)
    y_min_best = y - K.one_hot(best, K.shape(y)[-1])
    second_best = K.argmax(y_min_best, axis=-1)

    best_value = K.max(y, axis=-1)
    second_best_value = K.max(y_min_best, axis=-1)
    return K.stack(
        [K.stack([K.cast(best, 'float32'), best_value]), K.stack([K.cast(second_best, 'float32'), second_best_value])])


def get_full_model(name, inputs, get_model_fn):
    model_outputs = []
    for i in range(0, len(inputs)):
        _, model_output = get_model_fn(f'{name}_{model_names[i]}', inputs[i])
        model_outputs.append(model_output)

    _, main_output = sequential([
        Add(name=f'votes_sum_{name}')(model_outputs),
        Lambda(lambda x: x / len(inputs), name=f"norm_{name}"),
        Lambda(lambda x: get_2_max(x), name=f"argmax_{name}")
    ])

    return main_output


def get_release_model(inputs):
    _, output = get_model('release', inputs[-1], 5)
    return Lambda(lambda x: get_2_max(x), name=f"argmax_release")(output)


def recurring_layer_multiplexer(y):
    multilayer_pred, prohibitory, danger, direction, release, red_surface = y[0], y[1], y[2], y[3], y[4], y[5]
    prohib_map = K.constant([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 43], 'int32')
    danger_map = K.constant([11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 43], 'int32')
    direction_map = K.constant([33, 34, 35, 36, 37, 38, 39, 40, 43], 'int32')
    release_map = K.constant([6, 32, 41, 42, 43], 'int32')
    red_surface_map = K.constant([14, 17, 43], 'int32')

    multilayer_indices = multilayer_pred

    best_pred = K.switch(K.greater(multilayer_pred[0][0], 4),
                         18 - K.cast(multilayer_pred[0][0], 'int32'),
                         K.switch(K.equal(multilayer_indices[0][0], 0),
                                  K.gather(prohib_map, K.cast(prohibitory[0][0], 'int32')),
                                  K.switch(K.equal(multilayer_indices[0][0], 1),
                                           K.gather(danger_map, K.cast(danger[0][0], 'int32')),
                                           K.switch(K.equal(multilayer_indices[0][0], 2),
                                                    K.gather(direction_map, K.cast(direction[0][0], 'int32')),
                                                    K.switch(K.equal(multilayer_indices[0][0], 3),
                                                             K.gather(release_map, K.cast(release[0][0], 'int32')),
                                                             K.gather(red_surface_map,
                                                                      K.cast(red_surface[0][0], 'int32'))
                                                             )))))

    return best_pred
    best_pred_conf = K.switch(K.greater(multilayer_pred[0][0], 4),
                              multilayer_pred,
                              K.switch(K.equal(multilayer_indices[0][0], 0),
                                       prohibitory,
                                       K.switch(K.equal(multilayer_indices[0][0], 1),
                                                danger,
                                                K.switch(K.equal(multilayer_indices[0][0], 2),
                                                         direction,
                                                         K.switch(K.equal(multilayer_indices[0][0], 3),
                                                                  release,
                                                                  red_surface
                                                                  )))))

    return K.switch(K.equal(best_pred, 43),
                    K.switch(K.greater(best_pred_conf[0][1], multilayer_pred[0][1]),
                             K.switch(K.equal(multilayer_indices[1][0], 0),
                                      K.gather(prohib_map, K.cast(prohibitory[1][0], 'int32')),
                                      K.switch(K.equal(multilayer_indices[1][0], 1),
                                               K.gather(danger_map, K.cast(danger[1][0], 'int32')),
                                               K.switch(K.equal(multilayer_indices[1][0], 2),
                                                        K.gather(direction_map, K.cast(direction[1][0], 'int32')),
                                                        K.switch(K.equal(multilayer_indices[1][0], 3),
                                                                 K.gather(release_map, K.cast(release[1][0], 'int32')),
                                                                 K.gather(red_surface_map,
                                                                          K.cast(red_surface[1][0], 'int32'))
                                                                 )))),
                             K.switch(K.equal(multilayer_indices[0][0], 0),
                                      K.gather(prohib_map, K.cast(prohibitory[1][0], 'int32')),
                                      K.switch(K.equal(multilayer_indices[0][0], 1),
                                               K.gather(danger_map, K.cast(danger[1][0], 'int32')),
                                               K.switch(K.equal(multilayer_indices[0][0], 2),
                                                        K.gather(direction_map, K.cast(direction[1][0], 'int32')),
                                                        K.switch(K.equal(multilayer_indices[0][0], 3),
                                                                 K.gather(release_map, K.cast(release[1][0], 'int32')),
                                                                 K.gather(red_surface_map,
                                                                          K.cast(red_surface[1][0], 'int32'))
                                                                 )))),
                             ),
                    best_pred
                    )