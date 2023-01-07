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
    crossover_point = random.randint(0, len(weights2))
    weights2[crossover_point:] = weights1[crossover_point:]
    model2.set_weights(weights2)

    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model2


def cross_over_into_new(model1, model2):
    new_model = build_model()
    new_weights = new_model.get_weights()
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()

    crossover_point = random.randint(0, len(new_weights))

    new_weights[crossover_point:] = weights2[crossover_point:]
    new_weights[:crossover_point] = weights1[:crossover_point]

    new_model.set_weights(new_weights)
    new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return new_model


def build_small_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=8, activation='relu', input_shape=(8,)))
    model.add(keras.layers.Dense(units=16, activation='relu', ))
    model.add(keras.layers.Dense(units=16, activation='relu', ))
    model.add(keras.layers.Dense(units=4, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_random_model_small():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=8, activation='relu', input_shape=(8,)))
    model.add(keras.layers.Dense(units=16, activation='relu', kernel_initializer=initializers.random_normal))
    model.add(keras.layers.Dense(units=16, activation='relu', kernel_initializer=initializers.random_normal))
    model.add(keras.layers.Dense(units=4, activation='softmax', kernel_initializer=initializers.random_normal))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def small_crossover_into_one(model1, model2):
    new_model = build_small_model()
    random_model = get_random_model_small()
    random_weights = random_model.get_weights()
    new_weights = new_model.get_weights()
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()

    crossover_point = random.randint(0, len(new_weights))

    new_weights[crossover_point:] = weights1[crossover_point:]
    new_weights[:crossover_point] = weights2[:crossover_point]

    start_index = random.randint(0, len(new_weights))
    for i in range(start_index, start_index + int(len(new_weights) * 0.1)):
        new_weights[i] = random_weights[i]

    new_model.set_weights(new_weights)
    new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return new_model


def get_small_raw():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=8, activation='relu', input_shape=(8,)))
    model.add(keras.layers.Dense(units=16, activation='relu', ))
    model.add(keras.layers.Dense(units=16, activation='relu', ))
    model.add(keras.layers.Dense(units=8, activation='relu', ))
    model.add(keras.layers.Dense(units=4, activation='softmax'))
    return model


def local_compile(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def new_mutant_10_percent(model1):
    new_model = get_random_model_small()
    new_weights = new_model.get_weights()
    weights1 = model1.get_weights()

    start_index = random.randint(0, len(weights1) -1)
    for i in range(start_index, min(len(weights1)-1,  int(start_index + 0.1 * len(weights1)-1))):
        if random.randrange(0, 1, 1) == 1:
            weights1[i] = weights1[i] + new_weights[i]
        else:
            weights1[i] = weights1[i] - new_weights[i]
    new_model.set_weights(weights1)
    new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return new_model
