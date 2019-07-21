import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from util.plot import plot_training_results, plot_confusion_matrix
from util.data_util import get_data
from util.model import save_model

"""Data preparation"""
data_path = "./data/train_data.csv"
data_headers = ["polarity", "id", "date", "query", "user", "text"]

train_size = 100000
test_size = train_size * 0.2
batch_size = 200
train_epochs = 30

x_test, y_test, x_train, y_train = get_data(data_path, train_size, test_size, data_headers, skip_rows=1)

X = pd.concat([x_test, x_train])
Y = pd.concat([y_test, y_train])

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X['text'])

X = tokenizer.texts_to_sequences(X['text'])
Y = pd.get_dummies(Y['polarity'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
x_train = pad_sequences(x_train, maxlen=96)
x_test = pad_sequences(x_test, maxlen=96)

model = Sequential()
model.add(Embedding(20000, 128))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(64, dropout=0.35, recurrent_dropout=0.35))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=3,
                    verbose=2,
                    validation_data=(x_test, y_test))

plot_training_results(history)

y_pred = model.predict_classes(x_test)
plot_confusion_matrix(y_test[4], y_pred, [0, 1], ['Negative', 'Positive'], "/home/sabir/Documents/", normalize=True)

model_json = model.to_json()
save_model("./resources/lstm/lstm.json", ".../resources/lstm/lstm.h5", model_json, model)
