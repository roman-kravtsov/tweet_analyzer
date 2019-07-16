from classification.src.util.preprocessor import preprocess
from classification.src.util.features_extractor import get_word2vec_features, update_labels
from classification.src.util.data_util import get_data
from classification.src.util.model import get_model_word2vec_cnn, plot_training_results

from sklearn.metrics import confusion_matrix

"""Data preparation"""
data_path = "path_to_data"
data_headers = ["data", "headers", "here"]

train_size = 2000
test_size = 800
batch_size = 100
train_epochs = 10

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
matrix = confusion_matrix(test_labels, y_pred, labels=[0, 1])

print(matrix)

# Print incorrectly classified
# for i in range(len(x_test["text"])):
#     if test_labels_updated[i] != y_pred[i]:
#         print(i, test_labels_updated[i], y_pred[i], x_test.at[i, "text"])
