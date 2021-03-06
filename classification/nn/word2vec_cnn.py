from util.plot import plot_training_results
from util.preprocessor import preprocess
from util.features_extractor import get_word2vec_features, update_labels
from util.data_util import get_data
from util.model import get_model_word2vec_cnn

from sklearn.metrics import confusion_matrix, classification_report

"""Data preparation"""
data_path = "./data/train_data.csv"
data_headers = ["polarity", "id", "date", "query", "user", "text"]

train_size = 20000
test_size = train_size * 0.2
batch_size = 200
train_epochs = 30

x_test, y_test, x_train, y_train = get_data(data_path, train_size, test_size, data_headers, skip_rows=1)

x_test, y_test = preprocess(x_test, y_test)
x_train, y_train = preprocess(x_train, y_train)

features_train, features_test = get_word2vec_features(x_train, x_test)
train_labels, test_labels = update_labels(y_train, y_test)

"""Training and evaluation"""
model = get_model_word2vec_cnn(features_train[0].shape)

history = model.fit(features_train, train_labels, validation_data=(features_test, test_labels),
                    epochs=train_epochs, batch_size=batch_size)
# model.summary()

# Validation and training loss chart
plot_training_results(history)

# Confusion matrix
y_pred = model.predict_classes(features_test)

"""Metrics"""
matrix = confusion_matrix(test_labels, y_pred, labels=[0, 1])

print(matrix)
print(classification_report(test_labels, y_pred))

# Print incorrectly classified
# for i in range(len(x_test["text"])):
#     if test_labels_updated[i] != y_pred[i]:
#         print(i, test_labels_updated[i], y_pred[i], x_test.at[i, "text"])
