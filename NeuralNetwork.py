import tensorflow as tf
import pygame
import numpy as np
from tensorflow import keras


def build_model():
    # build the model
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(50 ,50)))
    model.add(keras.layers.Dense(units=100*100, activation='relu'))
    model.add(keras.layers.Dense(units=100, activation='relu'))
    model.add(keras.layers.Dense(units=100, activation='relu'))
    model.add(keras.layers.Dense(units=100, activation='relu'))
    model.add(keras.layers.Dense(units=100, activation='relu'))
    model.add(keras.layers.Dense(units=100, activation='relu'))
    model.add(keras.layers.Dense(units=100, activation='relu'))
    model.add(keras.layers.Dense(units=4, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
def test():
    model = build_model()

    array = np.random.randint(0,100,(50,50))
    model.predict(array)


