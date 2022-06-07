from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense


def model_1(input_shape):
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=input_shape, activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # flatten output and feed into dense layer
    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1, activation="sigmoid"))

    return model