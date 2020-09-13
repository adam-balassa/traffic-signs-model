from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


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