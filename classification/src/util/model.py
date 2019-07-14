from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization
from keras import optimizers

import matplotlib.pyplot as plt


def get_model_word2vec_cnn(input_shape):
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


def plot_training_results(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
