from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Lambda, Concatenate
from tensorflow.keras.models import Model


def sequential(layers):
    prev_layer = layers[0]
    for i in range(1, len(layers)):
        layer = layers[i]
        prev_layer = layer(prev_layer)
    return layers[0], prev_layer


def get_top_model():
    return sequential([
        Input(shape=(100, 100, 3), name=f'input'),
        Conv2D(50, kernel_size=3, padding='same', name=f'conv2d_top1'),
        Conv2D(50, kernel_size=3, padding='same', name=f'conv2d_top2', activation='relu'),
        MaxPool2D(pool_size=2, name=f"max_pool_top"),
        Conv2D(100, kernel_size=3, padding='same', name=f'conv2d_top3'),
        Conv2D(100, kernel_size=3, padding='same', name=f'conv2d_top4', activation='relu'),
        MaxPool2D(pool_size=2, name=f"max_pool_top2"),
        Conv2D(250, kernel_size=2, name=f'conv2d_top5', activation='relu')
    ])


def get_reg_model(input_layer):
    return sequential([
        input_layer,
        Conv2D(250, kernel_size=1, name=f'conv2d_reg', activation='relu'),
        Conv2D(4, kernel_size=1, name=f'reg_output', activation='relu')
    ])


def get_cls_model(input_layer):
    return sequential([
        input_layer,
        Conv2D(1, kernel_size=1, name=f'cls_output', activation='sigmoid')
    ])


def linear_output(output, name):
    return Lambda(lambda x: x, name=name)(output)


def get_full_model():
    input_layer, core = get_top_model()
    _, cls_out = get_cls_model(core)
    _, reg_out = get_reg_model(core)
    output = Concatenate(name="concat")([cls_out, reg_out])
    model = Model(inputs=input_layer, outputs=output)
    return model
