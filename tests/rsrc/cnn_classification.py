import numpy as np
import keras
from keras import layers

def cnn_method(x_train, x_test, y_train, y_test):
    num_classes = 3
    input_shape = (25, 50, 1)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(5, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(5,5)),
            # layers.Conv2D(34, kernel_size=(3, 3), activation="relu"),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()

    batch_size = 128
    epochs = 30
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    return model.evaluate(x_test, y_test, verbose=0)



def cnn_evaluation(x_nowarp, x, y):
    x_nowarp = x_nowarp.reshape((-1,25,50))
    x = x.reshape((-1, 25, 50))
    _, acc1 = cnn_method(x_nowarp[::2], x_nowarp[1::2], y[0::2], y[1::2])
    _, acc2 = cnn_method(x[::2], x[1::2], y[0::2], y[1::2])

    return None