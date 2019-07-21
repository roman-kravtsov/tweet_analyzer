from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization
from keras import optimizers

from keras.models import model_from_json


def get_model_word2vec_cnn(input_shape):
    """Returns a CNN model with given input_shape"""

    model = Sequential()

    model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv1D(32, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv1D(16, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = optimizers.Adam(lr=0.001, decay=1e-5)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def save_model(path_to_save, path_to_save_weights, model_json, model):
    """Saves model"""
    with open(path_to_save, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(path_to_save_weights)


def load_model(path_to_model, path_to_weights):
    """Loads model"""
    with open(path_to_model, "r") as json_file:
        loaded_model = model_from_json(json_file.read())
    loaded_model.load_weights(path_to_weights)
