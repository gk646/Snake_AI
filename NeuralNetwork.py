import random

import numpy as np
from keras import initializers
from tensorflow import keras


def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(50, 50)))
    model.add(keras.layers.Dense(units=50 * 50, activation='relu'))
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
    array = np.random.randint(0, 100, (50, 50))
    model.predict(array)


def cross_over(model1, model2):
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()

    crossover_point = random.randint(0, len(weights1))

    temp_weights = weights1[crossover_point:]
    weights1[crossover_point:] = weights2[crossover_point:]
    weights2[crossover_point:] = temp_weights

    model1.set_weights(weights1)
    model2.set_weights(weights2)

    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model1, model2


def get_random_model():
    new_model = keras.Sequential()
    new_model.add(keras.layers.Flatten(input_shape=(50, 50)))
    new_model.add(
        keras.layers.Dense(units=50 * 50, activation='relu', kernel_initializer=initializers.random_normal))
    new_model.add(
        keras.layers.Dense(units=100, activation='relu', kernel_initializer=initializers.random_normal))
    new_model.add(
        keras.layers.Dense(units=100, activation='relu', kernel_initializer=initializers.random_normal))
    new_model.add(
        keras.layers.Dense(units=100, activation='relu', kernel_initializer=initializers.random_normal))
    new_model.add(
        keras.layers.Dense(units=100, activation='relu', kernel_initializer=initializers.random_normal))
    new_model.add(
        keras.layers.Dense(units=100, activation='relu', kernel_initializer=initializers.random_normal))
    new_model.add(
        keras.layers.Dense(units=100, activation='relu', kernel_initializer=initializers.random_normal))
    new_model.add(keras.layers.Dense(units=4, activation='softmax'))

    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model


def half_random_model(model1):
    model2 = get_random_model()
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()
    crossover_point = random.randint(0, len(weights1))

    # Crossover the weights
    temp_weights = weights1[crossover_point:]
    weights1[crossover_point:] = weights2[crossover_point:]
    weights2[crossover_point:] = temp_weights

    # Set the weights of the models
    model1.set_weights(weights1)
    model2.set_weights(weights2)

    # Compile and train the models
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model1
