from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense


def get_model():
    model = Sequential()
    model.add(Conv2D(48, kernel_size=4, padding='same', activation='relu', input_shape=(48, 48, 3)))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(24, kernel_size=4, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(5, activation='sigmoid'))
    return model
