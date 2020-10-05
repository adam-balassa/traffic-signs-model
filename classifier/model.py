import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Add, Lambda
from tensorflow.keras.models import Model


def sequential(input_tensor, layers):
    prev_layer = input_tensor
    for layer in layers:
        prev_layer = layer(prev_layer)
    return prev_layer


def get_model():
    model = Sequential()
    model.add(Conv2D(40, kernel_size=7, padding='same', activation='relu', input_shape=(40, 40, 3)))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(20, kernel_size=4, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(10, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dense(43, activation='softmax'))  # hyperbolic tangent?
    return model


def extend_model(models):
    model_names = ['simple', 'stretch', 'eq', 'adeq']
    inputs = [Input(shape=(40, 40, 3), name=f"input_{name}") for name in model_names]

    model_outputs = [sequential(inputs[i], models[i].layers) for i in range(0, len(inputs))]

    votes_sum = Add(name="votes_sum")(model_outputs)
    votes = Lambda(lambda x: K.argmax(x, axis=-1), name="votes")(votes_sum)

    return Model(inputs=inputs, outputs=votes)
